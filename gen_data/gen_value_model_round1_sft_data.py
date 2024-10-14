import json
import os
import random


data = json.load(open("mcts_plan/data/5k_iter200.json", "r"))
sft_data = {}

TOO_MANY_STEPS = "Fail to sove the problem within limited steps!!!"
DO_NOT_FOLLOW_INSTRUCTION = "The response does not follow the instruction!!!"

def dfs(instruction, output, tree, node_idx):
    for i in range(3):
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION:
                continue
            if tree[child_node_idx]['solution']:
                solution = tree[child_node_idx]['solution'].replace('Detailed Implementation:', '').strip()
                cur_output = f"</plan>\n<solution>\n{solution}\n</solution>\n"
                if not sft_data.get(instruction):
                    sft_data[instruction] = []
                sft_data[instruction].append(output + [{"step": cur_output, "Q": tree[child_node_idx]['q_value']}])
            elif tree[child_node_idx]['step_plan']:
                cur_output = f"<step>\n{tree[child_node_idx]['step_plan'].strip()}\n</step>\n"
                output = output + [{"step": cur_output, "Q": tree[child_node_idx]['q_value']}]
                dfs(instruction, output, tree, child_node_idx)
                output = output[:-1]
        
for d in data:
    tree = d['plan_and_solve']
    instruction = f"<question>\n{d['question']}\n</question>\n"
    node_idx = "0"

    for i in range(5):
        child_node_idx = f"{node_idx}.{i}"
        if child_node_idx in tree.keys():
            if tree[child_node_idx]['final_answer'] == TOO_MANY_STEPS or tree[child_node_idx]['final_answer'] == DO_NOT_FOLLOW_INSTRUCTION:
                continue   
            elif tree[child_node_idx]['step_plan']:
                cur_step = f"<plan>\n<step>\n{tree[child_node_idx]['step_plan'].strip()}\n</step>\n"
                output = [{"step": cur_step, "Q": tree[child_node_idx]['q_value']}]
                dfs(instruction, output, tree, child_node_idx)


selected_data = []
for question in sft_data.keys():
    correct_solutions = [sft_data[question][i] for i in range(len(sft_data[question])) if sft_data[question][i][-1]['Q'] == 1]
    incorrect_solutions = [sft_data[question][i] for i in range(len(sft_data[question])) if sft_data[question][i][-1]['Q'] == -1]
    selected = []
    # if len(correct_solutions) >= 15:
    #     selected += random.sample(correct_solutions, 15)
    # else:
    #     selected += correct_solutions
    # if len(incorrect_solutions) >= 15:
    #     selected += random.sample(incorrect_solutions, 15)
    # else:
    #     selected += incorrect_solutions  
    selected += correct_solutions
    selected += incorrect_solutions
    for i in range(len(selected)):
        selected_data.append(
            {
                "instruction": question,
                "output": selected[i]
            }
        )

random.seed(42)
random.shuffle(selected_data)
os.makedirs("sft_value_head/data", exist_ok=True)
json.dump(selected_data, open("sft_value_head/data/value_model_round1_sft_data.json", "w"), indent=2)