import gc
import os
import json
from typing import List, Optional
import torch
from datetime import datetime
from dataclasses import dataclass
from copy import deepcopy

from tqdm import tqdm
from omegaconf import OmegaConf

from vllm import LLM, SamplingParams, CompletionOutput
from transformers import AutoTokenizer
from openai import OpenAI

from pebble import ProcessPool
from concurrent.futures import TimeoutError
from functools import partial

from utils import set_seed, batch
from constant import TIMEOUT_SECONDS
from mcts.modeling_value_model import ValueModel
from prompt import FEWSHOT_INST, FEWSHOT_XML


@dataclass
class CustomRequestOutput:
    prompt: Optional[str] = None
    value_estimate: Optional[float] = None
    outputs: Optional[List[CompletionOutput]] = None

class Solver:
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.stop = OmegaConf.to_object(self.config.stop)
        self.llm = self.create_llm()
        self.max_solve_steps = self.config.iterations

        # For step save
        self.data = data
        self.llm_name = os.path.basename(config.policy_model_dir.rstrip('/'))


    def init_value_model(self):
        # GPUS = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        value_model = ValueModel.from_pretrained(
            self.config.value_model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        vhead_params = torch.load(os.path.join(self.config.value_model_dir, 'value_head.pth'))
        vhead_params_without_prefix = {key.replace('v_head.', ''): value for key, value in vhead_params.items()}
        value_model.v_head.load_state_dict(vhead_params_without_prefix)
        return value_model

    def create_llm(self):
        if self.config.seed is not None:
            set_seed(self.config.seed)

        self.value_model = self.init_value_model()
        self.sampling_params = {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "n": self.config.n_generate_samples,
            "stop": self.stop,
            "seed": self.config.seed,
            "logprobs": 1
        }

        self.policy_tokenizer = AutoTokenizer.from_pretrained(self.config.policy_model_dir)
        self.value_tokenizer = AutoTokenizer.from_pretrained(self.config.value_model_dir)
        self.value_tokenizer.pad_token = self.value_tokenizer.eos_token
        self.value_tokenizer.padding_side = "left"

        self.fewshot = FEWSHOT_INST if self.config.round == 1 else FEWSHOT_XML

        return partial(
            engine_generate,
            value_model=self.value_model,
            policy_tokenizer=self.policy_tokenizer,
            value_tokenizer=self.value_tokenizer,
            fewshot=self.fewshot
        )
    
    def generate_preprocess(self, trees):
        prompts = []
        valid_trees = []
        invalid_trees = []

        for tree in trees:
            if tree.should_generate_next():
                tree_prompt = tree.create_prompt()
                prompts.append(tree_prompt)
                valid_trees.append(tree)
            else:
                invalid_trees.append(tree)
        return prompts, valid_trees, invalid_trees


    @staticmethod
    def processor(tree, request_output):
        tree.generate_next_step(request_output)
        return tree

    def expansion_and_backpropagation(self, request_outputs, valid_trees):
        post_trees = []
        with ProcessPool(max_workers=min(len(valid_trees), os.cpu_count())) as pool:
            future = pool.map(self.__class__.processor, valid_trees, request_outputs, timeout=TIMEOUT_SECONDS)
            iterator = future.result()
        
            if len(valid_trees) > 100:  
                progress_bar = tqdm(total=len(valid_trees), desc="Expanding and Backpropagating")  
            else:  
                progress_bar = None 

            while True:
                try:
                    result = next(iterator)
                    post_trees.append(result)
                except StopIteration:
                    break
                except TimeoutError as error:
                    post_trees.append(None)
                    print(f"TimeoutError: {error}")
                except SystemExit as error:
                    post_trees.append(None)
                    print(f"SystemExit: {error}")
                except Exception as error:
                    post_trees.append(None)
                    print(f"Exception in expansion and backpropagation: {error}")
                if progress_bar is not None:
                    progress_bar.update(1) 
            
            if progress_bar is not None:
                progress_bar.close() 

        # update solvers
        assert len(valid_trees) == len(post_trees), f"Data is not matched, {len(valid_trees)} vs {len(post_trees)}."
        updated_trees = [
            post_solver if post_solver is not None else valid_solver
            for post_solver, valid_solver in zip(post_trees, valid_trees)
        ]
        return updated_trees
    
    def solve(self, trees):
        for step in tqdm(range(self.max_solve_steps), desc="Step Processing"):
            prompts, valid_trees, invalid_trees = self.generate_preprocess(trees)
            if len(valid_trees) < 1:
                break
            
            if step == 0:
                self.sampling_params['n'] = 5
            else:
                self.sampling_params['n'] = self.config.n_generate_samples

            outputs= self.llm(prompts, self.sampling_params)

            # expansion and backpropagation    
            valid_trees = self.expansion_and_backpropagation(outputs, valid_trees)

            # selection
            for tree in valid_trees:
                tree.select_next_step()

            trees = invalid_trees + valid_trees

            if (step + 1) % 10 == 0:
                saved_json_file = f"{self.config.qaf}.mcts.{self.llm_name}.{datetime.now().strftime('%Y%m%d%H%M%S')}_iter{step+1}.json"
                step_data = []
                step_plan_and_solve = self.output(trees)
                for d in self.data:
                    d = deepcopy(d)
                    d['plan_and_solve'] = step_plan_and_solve[d['question']]
                    step_data.append(d)
                json.dump(step_data, open(saved_json_file, "w"), ensure_ascii=False, indent=4)

        return self.output(trees)

    def output(self, trees):
        outputs = {}
        for tree in trees:
            outputs[tree.question] = tree.return_states()
            
        return outputs


        
def single_request(prompt, sampling_params, fewshot):
    try:
        client = OpenAI(
            base_url="http://0.0.0.0:8000/v1",
            api_key="api-key",
        )
        prompt = fewshot + prompt
        completion = client.completions.create(
            model="policy_model",
            prompt=prompt,
            **sampling_params
        )
        return completion.choices
    except Exception as e:
        print(e)
        return None

def engine_generate(
        prompts, 
        sampling_params, 
        value_model, 
        policy_tokenizer, 
        value_tokenizer, 
        fewshot
    ):
    value_estimates = []
    batch_size = 32

    for prompt_batch in tqdm(batch(prompts, batch_size), total=len(prompts) // batch_size + 1, desc="Value model generating..."):
        inputs = value_tokenizer(
            prompt_batch, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024,
            add_special_tokens=False
        )
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                _, _, values = value_model(**inputs)

        torch.cuda.empty_cache()
        values = values.detach().cpu().to(torch.float32).tolist()
        value_estimates.extend(values)

    completion_choices = []
    max_workers = min(len(prompts), os.cpu_count()) 

    with ProcessPool(max_workers=max_workers) as pool:
        future = pool.map(
            single_request, 
            prompts, 
            [sampling_params] * len(prompts),
            [fewshot] * len(prompts),
        )
        iterator = future.result()

        if len(prompts) > 10:
            progress_bar = tqdm(total=len(prompts), desc="Policy model generating...")
        else:
            progress_bar = None

        while True:
            try:
                result = next(iterator)
                completion_choices.append(result)
            except StopIteration:
                break
            except Exception as error:
                completion_choices.append(None)
                print(f"Exception in engine generate: {error}")

            if progress_bar is not None:
                progress_bar.update(1)

        if progress_bar is not None:
            progress_bar.close()

    request_outputs = []
    n_answers = sampling_params['n']
    for i in range(len(prompts)):
        completion_choice = completion_choices[i]
        if completion_choice is None:
            request_outputs.append(CustomRequestOutput())
            continue
        else:
            completion_outputs = []
            for j in range(n_answers):
                text = completion_choice[j].text
                token_ids = policy_tokenizer.convert_tokens_to_ids(completion_choice[j].logprobs.tokens)
                cumulative_logprob = sum(completion_choice[j].logprobs.token_logprobs)
                completion_outputs.append(
                    CompletionOutput(
                        index=j,
                        text=text,
                        token_ids=token_ids,
                        cumulative_logprob=cumulative_logprob,
                        logprobs=completion_choice[j].logprobs.token_logprobs,
                    )
                )
            request_outputs.append(
                CustomRequestOutput(
                    prompt=prompts[i],
                    value_estimate=value_estimates[i],
                    outputs=completion_outputs
                )
            )
    return request_outputs