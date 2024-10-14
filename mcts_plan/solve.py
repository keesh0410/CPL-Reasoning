import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
from tqdm import tqdm
from datetime import datetime

import argparse
from omegaconf import OmegaConf

from utils import load_qaf, batch
from mcts import MCTSTree, Solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--custom_cfg', type=str, required=True)
    parser.add_argument('--qaf', type=str, required=True)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.custom_cfg)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    config = OmegaConf.merge(config, OmegaConf.create(vars(args)))
    print(config)

    llm_name = os.path.basename(config.policy_model_dir.rstrip('/'))

    data = load_qaf(args.qaf)

    solver = Solver(config=config, data=data)
    
    saved_jsonl_file = f"{args.qaf}.mcts.{llm_name}.{datetime.now().strftime('%Y%m%d%H%M%S')}.jsonl"
    
    
    with open(saved_jsonl_file, "w") as writer:
        for cur_data in tqdm(batch(data, -1), desc="Main Processing"):
            trees = [MCTSTree(config=config, question=d['question'], ground_truth=d['answer']) for d in cur_data]
            plan_and_solve = solver.solve(trees)
            for d in cur_data:
                d['plan_and_solve'] = plan_and_solve[d['question']]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()


    