import os
import re
import random
import torch
import numpy as np
import json

from math_evaluation import is_equiv
from constant import *


def load_qaf(filename):
    if filename.endswith(".json"):
        with open(filename, "r") as f:
            data = json.load(f)
    elif filename.endswith(".jsonl"):
        data = []
        with open(filename, "r") as f:
            lines = f.readlines()
        for line in lines:
            data.append(json.loads(line))
    else:
        raise ValueError(f"Unrecognized file format: {filename}")
    return data

def set_seed(seed: int = 1024):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

def batch(iterable, n=-1):
    l = len(iterable)
    if n < 0:
        n = l
    for ndx in range(0, l, n):
        yield iterable[ndx: min(ndx + n, l)]


def extract_boxed(output):
    lidx = output.rfind("\\boxed{")
    if lidx > 0:
        lidx += len("\\boxed{")
        ridx = output.rfind("}")
        return output[lidx: ridx]
    return None

def round1_prompt_wrap(question, partial_solution, config):
    step_delim = config.step_delim
    if partial_solution:
        inputs = f"Question: {question}\n\nPlans:\n{partial_solution}{step_delim}"
    else:
        inputs = f"Question: {question}\n\nPlans:\n"

    return inputs

def prompt_wrap(question, partial_solution, config):
    if partial_solution:
        inputs = f"<question>\n{question}</question>\n{partial_solution}{config.step_delim}"
    else:
        inputs = f"<question>\n{question}</question>\n"
    return inputs

def round1_step_unwrap(text):
    parser_result = {
        "step_plan": "",
        "solution": "",
        "final_answer": ""
    }

    plan_match = bool(re.match(r'^Plan \d+:\s*\S', text.strip()))
    solution_match = 'Detailed Implementation:' in text.strip()

    if solution_match and not plan_match:
        solution = text.strip()
        final_answer = extract_boxed(solution)
        if final_answer:
            parser_result["final_answer"] = final_answer
            parser_result["solution"] = solution
        else:
            parser_result["final_answer"] = DO_NOT_FOLLOW_INSTRUCTION
    elif plan_match and not solution_match:
        parser_result["step_plan"] = text.strip()
    else:
        parser_result["final_answer"] = DO_NOT_FOLLOW_INSTRUCTION

    return text, parser_result

def step_unwrap(text):
    parser_result = {
        "step_plan": "",
        "solution": "",
        "final_answer": ""
    }
    plan_pattern = r"Plan \d+"
    plan_matches = re.findall(plan_pattern, text)
    
    plan_match = bool(plan_matches)
    solution_match = "<solution>" in text

    if solution_match and not plan_match:
        solution = text
        final_answer = extract_boxed(solution)
        if final_answer:
            parser_result["final_answer"] = final_answer
            parser_result["solution"] = solution + "</solution>"
        else:
            parser_result["final_answer"] = DO_NOT_FOLLOW_INSTRUCTION
    elif plan_match and not solution_match:
        multi_plans = False if len(plan_matches) == 1 else True

        # consider Plan x+1 mention Plan x, such as Plan 2: calculate the x in Plan 1 
        if len(plan_matches) >= 2:
            first_x = int(plan_matches[0].split()[1])  # "Plan 2" -> 2
            second_x = int(plan_matches[1].split()[1])  # "Plan 1" -> 1
            if first_x > second_x:  
                multi_plans = False
        if not multi_plans:
            parser_result["step_plan"] = text
        else:
            parser_result["final_answer"] = DO_NOT_FOLLOW_INSTRUCTION
    else:
        parser_result["final_answer"] = DO_NOT_FOLLOW_INSTRUCTION
    
    return text, parser_result


def remove_single_dollar(s):
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1]
    return s

def math_is_equiv(grt, prd):
    prd = remove_single_dollar(prd)
    if isinstance(grt, list):
        for g in grt:
            if is_equiv(remove_single_dollar(g), prd):
                return True
        return False
    else:
        return is_equiv(remove_single_dollar(grt), prd)


                




    


