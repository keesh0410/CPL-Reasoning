import torch
import os.path as osp
from huggingface_hub import HfApi, hf_hub_download
from trl.models.modeling_value_head import ValueHead
from transformers import AutoConfig
import os
import shutil

repo_id = "deepseek-ai/deepseek-math-7b-base"

# policy model
api = HfApi()
repo_files = api.list_repo_files(repo_id=repo_id)

policy_model_save_path = "./models/policy_model_round1"
os.makedirs(policy_model_save_path, exist_ok=True)

for file_name in repo_files:
    hf_hub_download(repo_id=repo_id, filename=file_name, local_dir=policy_model_save_path)

# value model
value_model_save_path = "./models/value_model_round1"
if not os.path.exists(value_model_save_path):
    shutil.copytree(policy_model_save_path, value_model_save_path)

## initialize
config = AutoConfig.from_pretrained(repo_id)
v_head = ValueHead(config)
v_head.summary.weight.data.normal_(mean=0.0, std=config.initializer_range)
v_head.summary.bias.data.zero_()

## save file
v_head_state_dict = v_head.state_dict()
v_head_state_dict_with_prefix = {f'v_head.{k}': v for k, v in v_head_state_dict.items()}
torch.save(v_head_state_dict_with_prefix, osp.join(value_model_save_path, "value_head.pth"))



