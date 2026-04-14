# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json

import fire
import numpy as np
import vllm
from jinja2 import Template

from datasets import load_from_disk
from math_grader import (answer_tag_reward_fn,
                                            boxed_reward_fn)
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


def apply_qwen_math_template(question: str):
    return (
        "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
def apply_qwen_base_math_template(question:str):
    return (
        "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + question
        + " Let's think step by step and output the final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_qwen_boxed_math_template(question:str):
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        + question
        + "\nPlease reason step by step, and put your final answer within \\boxed{{}}.<|im_end|>\n<|im_start|>assistant\n"
    )

def apply_r1_template(question: str):
    return (
        "A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. "
        "The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\nUser: "
        + question
        + "\nAssistant: <think>"
    )

def apply_our_template(question: str):
    return (
        "<|im_start|>system\nA conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.\n<|im_end|>\n<|im_start|>user\n"
        + question
        + "<|im_end|>\n<|im_start|>assistant\n<think>"
    )

# The following two templates are used to evaluate baselines from other projects.
def apply_prime_zero_template(question: str):
    """https://huggingface.co/PRIME-RL/Eurus-2-7B-PRIME-Zero"""
    question = question + "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
    return f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {question}. Assistant:"


def apply_open_reasoner_zero_template(question: str):
    "https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/blob/e008f6d95f0b9a0e992f6b8bac912515b50a4634/playground/zero_setting_base.py"
    prompt_template_jinja = """\
{{bos_token}}A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. \
The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>. User: {{prompt}}
Assistant: <think>\
"""
    prompt_instruction_template_jinja = """\
You must put your answer inside <answer> </answer> tags, i.e., <answer> answer here </answer>. And your final answer will be extracted automatically by the \\boxed{} tag.
This is the problem:
{{prompt}}
"""
    prompt_instruction_template = Template(prompt_instruction_template_jinja)
    prompt_instruction = prompt_instruction_template.render(prompt=question)
    prompt_template = Template(prompt_template_jinja)
    return prompt_template.render(bos_token="", prompt=prompt_instruction)


def main(
    model_name: str = "Qwen/Qwen2.5-Math-1.5B",
    # tasks: list = ["aime", "amc", "math", "minerva", "olympiad_bench"],
    tasks: list = ["aime"],
    # tasks: list = ["amc","minerva"],
    template: str = "qwen_math",
    dataset_name: str = "./datasets/evaluation_suite",
    temperatures: str = "0.0",
    top_p: float = 1,
    max_tokens: int = 3000,
    max_model_len: int = 4096,  # VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 for longer ones.
    n_samples: int = 16,
    max_test: int = 999999,
    save: bool = False,
    seeds: str = "0"
):

    sampling_params = vllm.SamplingParams(
                n=n_samples,
                temperature=0.6,
                top_p=top_p,
                max_tokens=max_tokens,
                logprobs=0,
                seed=0,
            )

    model = vllm.LLM(
        model_name,
        swap_space=32,
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )

    # if "prime" in model_name.lower():
    #     template = "prime-zero"
    # if "open-reasoner-zero" in model_name.lower():
    #     template = "open-reasoner-zero"

    if "instruct" in model_name.lower() and "instruct" not in template:
        input(
            f"{model_name}\n{template}\ninstruct model but not instruct template! continue?"
        )

    print("Using template:", template)
    if template in ["qwen_math", "no"]:
        math_reward_fn = boxed_reward_fn
        # sampling_params.stop = [
        #     "</s>",
        #     "<|im_end|>",
        #     "<|endoftext|>",
        #     "<|im_start|>",
        #     "\nUser:",
        # ]
        # sampling_params.stop_token_ids = [151645, 151643]
        if template == "qwen_math":
            apply_template = apply_qwen_math_template
        else:
            apply_template = lambda x: x
    elif template=="qwen_base":
        print("============================================="*2)
        print("Using Qwen_base Template for Inference...")
        print("============================================="*2)
        math_reward_fn = boxed_reward_fn
        apply_template = apply_qwen_base_math_template
    elif template=="our":
        math_reward_fn = boxed_reward_fn
        apply_template = apply_qwen_math_template
    elif template == "r1":
        math_reward_fn = answer_tag_reward_fn
        sampling_params.stop = ["</answer>"]
        sampling_params.include_stop_str_in_output = True
        apply_template = apply_r1_template
    elif template=="auto":
        if "prime" in model_name.lower():
            math_reward_fn = boxed_reward_fn
            apply_template = apply_prime_zero_template
        elif "open-reasoner-zero" in model_name.lower():
            from understand_r1_zero.math_grader import answer_tag_reward_fn_for_orz
            math_reward_fn = answer_tag_reward_fn_for_orz
            apply_template = apply_open_reasoner_zero_template
        elif "oat-zero" in model_name.lower():
            math_reward_fn = boxed_reward_fn
            apply_template = apply_qwen_math_template
        elif "simplerl" in model_name.lower():
            math_reward_fn = boxed_reward_fn
            apply_template = apply_qwen_boxed_math_template
    elif template == "llama-instruct":

        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [
                    {
                        "content": f"{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n\n",
                        "role": "user",
                    }
                ],
                tokenize=False,
                add_generation_prompt=True,
            )

    elif template == "r1d":  # r1-distill
        from transformers import AutoTokenizer

        math_reward_fn = boxed_reward_fn

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def apply_template(question):
            return tokenizer.apply_chat_template(
                [{"content": question, "role": "user"}],
                tokenize=False,
                add_generation_prompt=True,
            )

    else:
        raise ValueError

    results = {}
    avg_lens = {}
    max_lens = {}
    formatted = {}
    to_be_saved = []
    if type(seeds)==int:
        seeds=[seeds]
    else:
        seeds = seeds.split()
    if type(temperatures)==float:
        temperatures=[temperatures]
    else:
        temperatures = temperatures.split()
    for seed in seeds:
        for temperature in temperatures:
            sampling_params.temperature=float(temperature)
            sampling_params.seed=int(seed)
            print(f"seed:{sampling_params.seed},temperature:{sampling_params.temperature}")
            for task_name, dataset in load_from_disk(dataset_name).items():
                if task_name not in tasks:
                    continue
                prompts = dataset["problem"][:max_test]
                targets = dataset["answer"][:max_test]

                prompts = list(map(apply_template, prompts))
                # print(prompts[0])
                print("inference for ", task_name)
                outputs = model.generate(prompts, sampling_params)
                batch_scores = []
                batch_formatted = []
                batch_lengths = []
                for k in range(len(outputs)):
                    output = outputs[k]
                    gt_repeated = [targets[k]] * sampling_params.n
                    rewards, infos = [], []
                    for model_output, gt in zip([o.text for o in output.outputs], gt_repeated):
                        info, r = math_reward_fn(model_output, gt, fast=False)
                        rewards.append(r)
                        infos.append(info)
                    rewards = np.array(rewards)
                    batch_lengths.append([len(o.token_ids) for o in output.outputs])
                    batch_scores.append(rewards.mean())

                    if infos[0] is not {}:
                        batch_formatted.append(np.array([i["formatted"] for i in infos]).sum())

                    to_be_saved.append(
                        {
                            "task_name": task_name,
                            "prompt": output.prompt,
                            "gt": gt_repeated,
                            "model_output": [o.text for o in output.outputs],
                            "reward": [r for r in rewards],
                        }
                    )

                results[task_name] = float(np.mean(batch_scores))
                avg_lens[task_name] = float(np.mean(batch_lengths))
                if batch_formatted:
                    formatted[task_name] = np.mean(batch_formatted)
                max_lens[task_name] = int(np.max(batch_lengths))

            save_result={}
            save_result["results"]=results
            save_result["avg"]=np.mean(list(results.values()))
            save_result["avg_lens"]=avg_lens
            save_result["max_lens"]=max_lens
            save_result["formatted"]=formatted
            save_result=[save_result]
            print(results)
            print("avg:", np.mean(list(results.values())))
            print("avg_lens:", avg_lens)
            print("max_lens:", max_lens)
            print("formatted:", formatted)

            if save:
                fn = "./eval_results/" + model_name.replace("/actor/huggingface","").replace("/", "_") + f"seed{seed}"
                fn = f"{fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json"
                print(f"saving model outputs at {fn}")
                with open(fn,"w") as f:
                    json.dump(to_be_saved,f,indent=4)
                fn=fn.replace(".json","_final_result.json")
                with open(fn,"w") as f:
                    json.dump(save_result,f,indent=2)
            
            
fire.Fire(main)