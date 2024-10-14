import random
import numpy as np

from mcts.mcts_node import MCTSNode
from utils import (
    math_is_equiv,
    round1_prompt_wrap,
    round1_step_unwrap,
    prompt_wrap,
    step_unwrap
)
from constant import *


class MCTSTree:
    def __init__(self, config, question, ground_truth):
        super().__init__()
        self.config = config
        self.question = question
        self.ground_truth = ground_truth

        self.prompt_wrap = round1_prompt_wrap if self.config.round == 1 else prompt_wrap
        self.step_unwrap = round1_step_unwrap if self.config.round == 1 else step_unwrap

        self.root = self.create_node()
        self.root.state['text'] = self.prompt_wrap(self.question, partial_solution=None, config=self.config)

        self.select_next_step()


    def create_node(self, parent=None):
        return MCTSNode(parent=parent)
    
    def create_child(self, prompt, step_text, parser_result, parent, prior_prob, idx):
        child = self.create_node(parent=parent)
        child.tag = f"{parent.tag}.{idx}"
        child.depth = parent.depth + 1
        child.prior = prior_prob
        child.state['prompt'] = prompt
        child.state['text'] = step_text

        if child.depth > self.config.max_depth:
            child.is_terminal = True
            child.state['final_answer'] = TOO_MANY_STEPS
            self.eval_final_answer(child)
        elif parser_result["final_answer"] == DO_NOT_FOLLOW_INSTRUCTION:
            child.is_terminal = True
            child.state['final_answer'] = DO_NOT_FOLLOW_INSTRUCTION
            self.eval_final_answer(child)
        else:
            child.state.update(parser_result)
            if child.state['final_answer']:
                child.is_terminal = True
                self.eval_final_answer(child)
        
        parent.children.append(child)
    
    def select_next_step(self):
        node = self.root
        while node.has_children():
            next_node = self.select_child(node)
            if next_node is None:
                node.is_terminal = True
                break
            node = next_node

        self.current_node = node

    def select_child(self, node):
        best_value = -np.inf
        best_childs = []
        for child in node.children:
            if child.is_terminal:
                continue
            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]
        
        return random.choice(best_childs) if best_childs else None
    
    def generate_next_step(self, request_output):
        value_estimate = request_output.value_estimate
        if value_estimate is not None: 
            self.current_node.update_recursive(value_estimate, self.root)
            self.expand_node(request_output, self.current_node)   
        else: # input exceeds max tokens, output '' and None
            self.current_node.update_recursive(self.config.negative_reward, self.root)
            self.current_node.is_terminal = True

    def expand_node(self, request_output, node):
        completion_outputs = request_output.outputs
        prompt = request_output.prompt
        # Todo: deversity
        dedup_outputs = []
        dedup_leys = set()
        for output in completion_outputs:
            key = output.text.strip()
            if key not in dedup_leys:
                dedup_leys.add(key)
                dedup_outputs.append(output)
        
        completion_outputs = dedup_outputs
        for idx, output in enumerate(completion_outputs):
            prior_prob = np.exp(output.cumulative_logprob / len(output.token_ids))
            step_text, parser_result = self.step_unwrap(output.text)
            self.create_child(prompt, step_text, parser_result, node, prior_prob, idx)

    def should_generate_next(self):
        return not self.current_node.is_terminal and self.current_node.depth <= self.config.max_depth
    
    def collect_partial_solution(self, node):
        trajectory = []
        while node.parent:
            if node.state["text"]:
                trajectory.append(node.state["text"])
            node = node.parent
        return self.config.step_delim.join(reversed(trajectory))
    
    def create_prompt(self):
        partial_solution = self.collect_partial_solution(self.current_node)
        prompt = self.prompt_wrap(self.question, partial_solution, self.config)
        return prompt

    def eval_final_answer(self, node):
        if node.state["final_answer"] in [TOO_MANY_STEPS, DO_NOT_FOLLOW_INSTRUCTION]:
            node.update_recursive(self.config.negative_reward, self.root)
            return 
        else:
            correct = math_is_equiv(node.state["final_answer"], self.ground_truth)
            node.update_recursive(self.config.positive_reward if correct else self.config.negative_reward, self.root)

    def return_states(self):
        candidates = [self.root]
        states = {}
        while candidates:
            node = candidates.pop(0)
            if node.visit_count() > 0:
                states[node.tag] = node.state
                states[node.tag]['value'] = node.value
                states[node.tag]['prior'] = node.prior
                states[node.tag]['visit_count'] = node.visit_count()
                states[node.tag]['q_value'] = node.q_value()
                candidates.extend(node.children)
        return states

        
