import json
import random
from transformers import AutoTokenizer


data = json.load(open('mcts_plan/data/5k_iter200.json', 'r'))
dpo_data = []
TOO_MANY_STEPS = "Fail to sove the problem within limited steps!!!"
DO_NOT_FOLLOW_INSTRUCTION = "The response does not follow the instruction!!!"

def dfs(prefix, tree, node_idx):
    output_list =[]
    cur_q_value_list = []
    prefix_list = []
    need_dfs = []
    prefix += "</step>\n"
    is_solution = False
    for i in range(3):
        cur_need_dfs = False
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION:
                continue
            if tree[child_node_idx]['solution']:
                prefix_list.append(prefix + "</plan>\n<solution>\n")
                solution = tree[child_node_idx]['solution'].replace("Detailed Implementation:", "").strip()
                cur_output = f"{solution}"   
                is_solution = True 
            elif tree[child_node_idx]['step_plan']:
                cur_need_dfs = True
                prefix_list.append(prefix + "<step>\n")
                cur_output = tree[child_node_idx]['step_plan'].strip()
            output_list.append(cur_output)
            cur_q_value_list.append(tree[child_node_idx]['q_value'])
            if cur_need_dfs:
                need_dfs.append((prefix_list[-1] + cur_output, tree, child_node_idx))

    if len(cur_q_value_list) > 1 and max(cur_q_value_list) > 0 and min(cur_q_value_list) < 0 and not is_solution:
        bigger_than_zero_idx = [i for i in range(len(cur_q_value_list)) if cur_q_value_list[i] > 0]
        smaller_than_zero_idx = [i for i in range(len(cur_q_value_list)) if cur_q_value_list[i] < 0]
        for i in bigger_than_zero_idx:
            for j in smaller_than_zero_idx:
                if prefix_list[i] == prefix_list[j]:
                    dpo_data.append(
                        {
                            "instruction": prefix_list[i],
                            "chosen": output_list[i],
                            "rejected": output_list[j],
                            "q_chosen": cur_q_value_list[i],
                            "q_rejected": cur_q_value_list[j]
                        }
                    )
    elif len(cur_q_value_list) > 1 and max(cur_q_value_list) > 0 and min(cur_q_value_list) < 0 and is_solution:
        max_idx = cur_q_value_list.index(max(cur_q_value_list))
        min_idx = cur_q_value_list.index(min(cur_q_value_list))
        if prefix_list[max_idx] == prefix_list[min_idx]:
            dpo_data.append(
                {
                    "instruction": prefix_list[max_idx],
                    "chosen": output_list[max_idx],
                    "rejected": output_list[min_idx],
                    "q_chosen": cur_q_value_list[max_idx],
                    "q_rejected": cur_q_value_list[min_idx]
                }
            )
    for (prefix, tree, node_idx) in need_dfs:
        dfs(prefix, tree, node_idx)
    
   
for d in data:
    tree = d['plan_and_solve']
    prefix = f"<question>\n{d['question']}\n</question>\n<plan>\n<step>\n"
    node_idx = "0"

    output_list =[]
    cur_q_value_list = []
    need_dfs = []
    for i in range(5):
        cur_need_dfs = False
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION:
                continue
            elif tree[child_node_idx]['solution']:
                continue
            elif tree[child_node_idx]['step_plan']:
                cur_need_dfs = True
                cur_output = tree[child_node_idx]['step_plan'].strip()
            output_list.append(cur_output)
            cur_q_value_list.append(tree[child_node_idx]['q_value'])
            if cur_need_dfs:
                need_dfs.append((prefix + cur_output, tree, child_node_idx))

    if len(cur_q_value_list) > 1 and max(cur_q_value_list) > 0 and min(cur_q_value_list) < 0:
        bigger_than_zero_idx = [i for i in range(len(cur_q_value_list)) if cur_q_value_list[i] > 0]
        smaller_than_zero_idx = [i for i in range(len(cur_q_value_list)) if cur_q_value_list[i] < 0]
        for i in bigger_than_zero_idx:
            for j in smaller_than_zero_idx:
                dpo_data.append(
                    {
                        "instruction": prefix,
                        "chosen": output_list[i],
                        "rejected": output_list[j],
                        "q_chosen": cur_q_value_list[i],
                        "q_rejected": cur_q_value_list[j]
                    }
                )
    for (prefix, tree, node_idx) in need_dfs:
        dfs(prefix, tree, node_idx)

sw = 0.3
for d in dpo_data:
    if d['instruction'].endswith("<solution>\n"):
        d['q_chosen'] = d['q_chosen'] * sw
        d['q_rejected'] = d['q_rejected'] * sw

random.seed(42)
random.shuffle(dpo_data)
json.dump(dpo_data, open('apo/LLaMA-Factory-0.8.3/data/policy_model_round1_apo_data.json', 'w'), indent=4, ensure_ascii=False)