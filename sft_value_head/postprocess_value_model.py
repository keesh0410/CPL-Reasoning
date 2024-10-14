from trl import AutoModelForCausalLMWithValueHead
from transformers import AutoConfig, AutoTokenizer
import torch
import safetensors
import torch
import os

model = AutoModelForCausalLMWithValueHead.from_pretrained('deepseek-ai/deepseek-math-7b-base', torch_dtype=torch.bfloat16)

# 加载本地 checkpoint
local_state_dict = safetensors.torch.load_file("../models/value_model_round1_sft/model.safetensors")
local_state_dict = {k: v.to(torch.bfloat16) for k, v in local_state_dict.items()}

for name, param in model.named_parameters():
    if name in local_state_dict:
        continue
    else:
        is_identical = False
        print(f"{name} doesn't exist in checkpoint")
        break

model.load_state_dict(local_state_dict)

v_head = model.v_head
v_head_state_dict = v_head.state_dict()
v_head_state_dict_with_prefix = {f'v_head.{k}': v for k, v in v_head_state_dict.items()}

# save file
save_path = "../models/value_model_round2"
os.makedirs(save_path, exist_ok=True)

torch.save(v_head_state_dict_with_prefix, os.path.join(save_path, "value_head.pth"))
model.pretrained_model.save_pretrained(save_path)

config = AutoConfig.from_pretrained('deepseek-ai/deepseek-math-7b-base')
config.value_model = True
config.save_pretrained(save_path)

tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/deepseek-math-7b-base')
tokenizer.save_pretrained(save_path)


