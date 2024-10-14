import os
import random
import json
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    HfArgumentParser, 
    set_seed, 
)

from trl import SFTConfig, AutoModelForCausalLMWithValueHead

from custom_trainer import CustomTrainer
from data_collator import preprocess_value_dataset, VMDataCollatorForSeq2Seq
from constant import *

os.environ["WANDB_API_KEY"] = "" # PUR YOUR WANDB KEY
os.environ["WANDB_PROJECT"] = "sft_value_head" # PUT YOUR WANDB PROJECT



@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="model_name_or_path",metadata={"help": "the location of the SFT model name or path"})
    data_path: Optional[str] = field(default="data/value_model_round1_sft_data.json", metadata={"help": "the location of the data"})
    cutoff_len: Optional[int] = field(default=1024, metadata={"help": "the cutoff length"})
    preprocessing_num_workers: Optional[int] = field(default=16, metadata={"help": "the number of workers"})


def main():
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)

    set_seed(training_args.seed)

    # Load datasets
    ds = load_dataset("json", data_files=script_args.data_path)
    ds = ds["train"].train_test_split(test_size=0.1, seed=training_args.seed)


    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    columns = list(ds['train'].features)
    dataset = ds.map(
        preprocess_value_dataset,
        batched=True,
        num_proc=script_args.preprocessing_num_workers,
        remove_columns=columns,
        fn_kwargs=dict(
            tokenizer=AutoTokenizer.from_pretrained(script_args.model_name_or_path),
            cutoff_len=script_args.cutoff_len,
        ),
        desc="Preprocessing dataset",
    )

    for index in random.sample(range(len(dataset)), 1):
        print(f"Random example {index}:")
        print(f"Input: {tokenizer.decode(dataset['train'][index]['input_ids'], skip_special_tokens=True)}")
    

    # load model
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        use_cache=False if training_args.gradient_checkpointing else True,
    )

    data_collator = VMDataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=script_args.cutoff_len)
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()

if __name__ == "__main__":
    main()