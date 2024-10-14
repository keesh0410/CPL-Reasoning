import json
data = json.load(open('mcts_plan/data/15k_iter100.json', 'r'))


questions = []
answers = {}
TOO_MANY_STEPS = "Fail to sove the problem within limited steps!!!"
DO_NOT_FOLLOW_INSTRUCTION = "The response does not follow the instruction!!!"

def dfs(question, tree, node_idx, prefix):
    for i in range(3):
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION or tree[child_node_idx]['q_value'] == -1.0:
                continue
            if tree[child_node_idx]['solution'] and tree[child_node_idx]['q_value'] == 1.0:
                cur_output = tree[child_node_idx]['solution']
                answers[question].append(prefix + cur_output)    
            elif tree[child_node_idx]['step_plan']:
                cur_output = f"{tree[child_node_idx]['step_plan']}</step>\n"
                cur_prefix = prefix + cur_output
                dfs(question, tree, child_node_idx, cur_prefix)
        
for d in data:
    tree = d['plan_and_solve']
    questions.append(d['question'])
    answers[d['question']] = []
    node_idx = "0"

    for i in range(5):
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION or tree[child_node_idx]['q_value'] == -1.0:
                continue   
            elif tree[child_node_idx]['step_plan']:
                cur_prefix = f"{tree[child_node_idx]['step_plan']}</step>\n"
                dfs(d['question'], tree, child_node_idx, cur_prefix)

import random
sft_data = []
for q, a in answers.items():
    if len(a) > 4:
        selected_a = random.sample(a, 4)
        for sa in selected_a:
            sft_data.append(
                {
                    "instruction": f"<question>\n{q}\n</question>\n",
                    "input": "",
                    "output": sa
                }
            )
    elif len(a) > 0:
        for sa in a:
            sft_data.append(
                {
                    "instruction": f"<question>\n{q}\n</question>\n",
                    "input": "",
                    "output": sa
                }
            )

print(len(sft_data))       
import random
random.seed(42)
random.shuffle(sft_data)
json.dump(sft_data, open('apo/LLaMA-Factory-0.8.3/data/policy_model_round2_sft_data.json', 'w'), indent=4, ensure_ascii=False)