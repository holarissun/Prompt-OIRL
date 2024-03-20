import os
import pandas as pd
import re
from tqdm import tqdm
import time
import json
import torch
from typing import Optional
import fire
from llama import Llama


def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_examples_gsm8k(split):
    path = os.path.join("../data/", f"{split}.jsonl")
    examples = read_jsonl(path)

    for ex in examples:
        ex.update(question=ex["question"] + "\n")
        ex.update(answer=ex["answer"] + "<|endoftext|>")

    print(f"{len(examples)} {split} examples")
    return examples


ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def extract_answer(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_completion, gt_example):
    gt_answer = extract_answer(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer(model_completion) == gt_answer


class GSMDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, examples, loss_on_prefix=True):
        self.examples = examples
        self.qns = [ex["question"] for ex in self.examples]
        self.ans = [ex["answer"] for ex in self.examples]
        self.qns = tokenizer(self.qns, padding=False)
        self.ans = tokenizer(self.ans, padding=False)
        self.loss_on_prefix = loss_on_prefix
        self.max_len = max(
            [
                len(self.qns["input_ids"][i]) + len(self.ans["input_ids"][i])
                for i in range(len(self.examples))
            ]
        )
        print(f"Max tokens: {self.max_len}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qn_tokens = self.qns["input_ids"][idx]
        ans_tokens = self.ans["input_ids"][idx]
        pad_tokens = [0] * (self.max_len - len(qn_tokens) - len(ans_tokens))
        tokens = qn_tokens + ans_tokens + pad_tokens
        mask = (
            ([int(self.loss_on_prefix)] * len(qn_tokens))
            + ([1] * len(ans_tokens))
            + ([0] * len(pad_tokens))
        )
        tokens = torch.tensor(tokens)
        mask = torch.tensor(mask)
        return dict(input_ids=tokens, attention_mask=mask)


"""
torchrun --nproc_per_node 1 llama_step1_gen_offline.py \
    --ckpt_dir llama-2-7b-chat/ \
    --tokenizer_path tokenizer.model \
    --max_seq_len 512 --max_batch_size 8 --prompt_idx 0 --dataset_eval gsm8k
"""
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


def get_examples(dataset):
    df = pd.concat(
        [
            pd.read_csv(os.path.join("../data/", f"{dataset}/fold{fold_i}/train.csv"))
            for fold_i in range(5)
        ]
    )

    # Assuming "Numbers" is a string representation of comma-separated values
    # Convert them into list of numbers
    df["Numbers"] = df["Numbers"].str.split(" ")

    def insert_numbers(row):
        question = row["Question"]
        for i, num in enumerate(row["Numbers"]):
            placeholder = "number" + str(i)
            question = re.sub(placeholder, str(num), question)
        return question

    df["Question"] = df.apply(insert_numbers, axis=1)

    print(df.head())
    print(df.shape)
    return df["Question"], df["Answer"]


def test_answer(pred_str, ans_str):
    pattern = "\d*\.?\d+"
    pred = re.findall(pattern, pred_str)
    if len(pred) >= 1:
        pred = pred[-1]
        gold = re.findall(pattern, ans_str)
        if isinstance(gold, str):
            gold = gold
        elif isinstance(gold, list):
            gold = gold[-1]
        return float(pred) == float(gold)
    else:
        return False


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
    prompt_idx: int = 0,
    dataset_eval: str = None,
):
    PROMPT_LIST = [
        "The answer is:",
        "Let's think step by step: ",
        "Let’s work this out in a step by step way to be sure we have the right answer: ",
        "3 experts are discussing the question with a discussion, trying to solve it step by step, and make sure the result is correct: ",
        "Imagine three different experts are answering this question. All experts will write down 1 step of their thinking, then share it with the group. Then all experts will go on to the next step, etc. If any expert realises they're wrong at any point then they leave",
        "First, decompose the question into several sub-questions that needs to solve, and then solve each question step by step: ",
        "Approaching this logically, the steps to find the answer are: ",
        "Let's break this down into manageable steps to fully understand the problem: ",
        "Consider this as a puzzle, each piece contributing to the final answer. Let's place each piece, one by one: ",
        "Three scholars are analyzing this query from various perspectives, working collaboratively to build a comprehensive answer. Each contributes a step: ",
        "Let's solve this like a detective would solve a mystery, gathering clues and building up to the final solution step by step: ",
        "Imagine we're navigating a maze; each decision brings us closer to the center. Let's map our route: ",
        "Envision a round table meeting of expert problem solvers. Each participant suggests a step, building towards a consensus answer: ",
        "Like an architect constructing a building, let's design our answer carefully, layer by layer: ",
        "As if we are assembling a complex machine, let's put it together piece by piece: ",
        "Three wise philosophers are debating this question, each contributing a different aspect of the answer. Let's follow their discourse: ",
    ]
    ALIAS_LIST = [
        "nan",
        "CoT",
        "APE",
        "Dis",
        "ToT",
        "Decomp",
        "gpt4_0",
        "gpt4_1",
        "gpt4_2",
        "gpt4_3",
        "gpt4_4",
        "gpt4_5",
        "gpt4_6",
        "gpt4_7",
        "gpt4_8",
        "gpt4_9",
    ]
    prompt = PROMPT_LIST[prompt_idx]
    file_alias = ALIAS_LIST[prompt_idx]
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    DATASET = dataset_eval
    if DATASET == "svamp":
        Questions, Answers = get_examples("cv_svamp_augmented")
    elif DATASET == "mawps":
        Questions, Answers = get_examples("cv_mawps")
    start_time = time.time()
    if DATASET == "svamp":
        test_idx_starter = 15000
    elif DATASET == "mawps":
        test_idx_starter = 6000

    if prompt_idx >= 6:
        if DATASET != "gsm8k":
            for ii, (ques_i, ans_i) in tqdm(
                enumerate(
                    zip(
                        Questions.values[test_idx_starter:],
                        Answers.values[test_idx_starter:],
                    )
                )
            ):
                print("time elapsed:", time.time() - start_time)
                print("time per question:", (time.time() - start_time) / (ii + 1))
                print(
                    "estimated time left:",
                    (time.time() - start_time)
                    / (ii + 1)
                    * (len(Questions.values) - ii - 1),
                )
                print("ques_i:", ques_i, prompt)
                print("dtype", type(ques_i), type(prompt))
                dialogs = [
                    [
                        {"role": "user", "content": ques_i + prompt},
                    ],
                ]
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                for dialog, result in zip(dialogs, results):
                    for msg in dialog:
                        print(f">> {msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f">>>> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print("answer:", ans_i)
                    TF = test_answer(result["generation"]["content"], str(ans_i))
                    print("TF:", TF)

                    # save results to file
                    with open(
                        f"results/llama_{DATASET}_answers_list_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{ii,result['generation']['content']}\n")
                        # insert a blank line
                        fd.write("\n====================\n")

                    with open(
                        f"results/llama_{DATASET}_corr_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{(ii,TF)}\n")

                    print("\n==================================\n")
        else:
            test_examples = get_examples_gsm8k("test")[:1000]
            for ii, example in tqdm(enumerate(test_examples)):
                ques_i, ans_i = example["question"], example["answer"]
                print("time elapsed:", time.time() - start_time)
                print("time per question:", (time.time() - start_time) / (ii + 1))
                print(
                    "estimated time left:",
                    (time.time() - start_time)
                    / (ii + 1)
                    * (len(test_examples) - ii - 1),
                )
                print("ques_i:", ques_i, prompt)
                print("dtype", type(ques_i), type(prompt))
                dialogs = [
                    [
                        {"role": "user", "content": ques_i + prompt},
                    ],
                ]
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                for dialog, result in zip(dialogs, results):
                    for msg in dialog:
                        print(f">> {msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f">>>> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print("answer:", ans_i)
                    TF = test_answer(result["generation"]["content"], str(ans_i))
                    print("TF:", TF)

                    # save results to file
                    with open(
                        f"results/llama_{DATASET}_answers_list_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{ii,result['generation']['content']}\n")
                        # insert a blank line
                        fd.write("\n====================\n")

                    with open(
                        f"results/llama_{DATASET}_corr_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{(ii,TF)}\n")
                    print("\n==================================\n")

    else:
        if DATASET != "gsm8k":
            for ii, (ques_i, ans_i) in tqdm(
                enumerate(zip(Questions.values, Answers.values))
            ):
                print("time elapsed:", time.time() - start_time)
                print("time per question:", (time.time() - start_time) / (ii + 1))
                print(
                    "estimated time left:",
                    (time.time() - start_time)
                    / (ii + 1)
                    * (len(Questions.values) - ii - 1),
                )
                print("ques_i:", ques_i, prompt)
                print("dtype", type(ques_i), type(prompt))
                dialogs = [
                    [
                        {"role": "user", "content": ques_i + prompt},
                    ],
                ]
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                for dialog, result in zip(dialogs, results):
                    for msg in dialog:
                        print(f">> {msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f">>>> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print("answer:", ans_i)
                    TF = test_answer(result["generation"]["content"], str(ans_i))
                    print("TF:", TF)

                    # save results to file
                    with open(
                        f"results/llama_{DATASET}_answers_list_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{ii,result['generation']['content']}\n")
                        # insert a blank line
                        fd.write("\n====================\n")

                    with open(
                        f"results/llama_{DATASET}_corr_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{(ii,TF)}\n")

                    print("\n==================================\n")
        else:
            test_examples = get_examples_gsm8k("train")
            for ii, example in tqdm(enumerate(test_examples)):
                ques_i, ans_i = example["question"], example["answer"]
                print("time elapsed:", time.time() - start_time)
                print("time per question:", (time.time() - start_time) / (ii + 1))
                print(
                    "estimated time left:",
                    (time.time() - start_time)
                    / (ii + 1)
                    * (len(test_examples) - ii - 1),
                )
                print("ques_i:", ques_i, prompt)
                print("dtype", type(ques_i), type(prompt))
                dialogs = [
                    [
                        {"role": "user", "content": ques_i + prompt},
                    ],
                ]
                results = generator.chat_completion(
                    dialogs,  # type: ignore
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )

                for dialog, result in zip(dialogs, results):
                    for msg in dialog:
                        print(f">> {msg['role'].capitalize()}: {msg['content']}\n")
                    print(
                        f">>>> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
                    )
                    print("answer:", ans_i)
                    TF = test_answer(result["generation"]["content"], str(ans_i))
                    print("TF:", TF)

                    # save results to file
                    with open(
                        f"results/llama_{DATASET}_answers_list_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{ii,result['generation']['content']}\n")
                        # insert a blank line
                        fd.write("\n====================\n")

                    with open(
                        f"results/llama_{DATASET}_corr_{file_alias}.txt", "a"
                    ) as fd:
                        fd.write(f"{(ii,TF)}\n")

                    print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
