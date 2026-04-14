#!/usr/bin/env python
# encoding: utf-8
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import fire
from glob import glob
from collections import defaultdict

# python convert_ckpt.py /path/to/fsdp_checkpoint allenai/OLMo-2-1124-7B-SFT /path/to/output_model

def main(fsdp_checkpoint_path, base_model_name_or_path, output_path):
    state_dict = defaultdict(list)

    world_size = 4
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        print('loading', filepath)
        this_state_dict = torch.load(filepath, weights_only=False)
        for key, value in this_state_dict.items():
            state_dict[key].append(value.to_local())

    for key in state_dict:
        state_dict[key] = torch.cat(state_dict[key], dim=0)

    # Load config and tokenizer from the original base model instead of the output path
    config = AutoConfig.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config)
    model.load_state_dict(state_dict)

    #for filepath in glob(f'{fsdp_checkpoint_path}/model_*.pt'):
    #    part_state_dict = torch.load(filepath)
    #    model.load_state_dict(part_state_dict)

    model.save_pretrained(output_path, max_shard_size="10GB")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)


if __name__ == "__main__":
    fire.Fire(main)