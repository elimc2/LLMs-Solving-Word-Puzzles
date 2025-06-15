#!/usr/bin/env python3
"""
build_word_puzzle_sft.py
Create train/validation splits in the ChatML/`messages` format required by
open_r1's SFT script.
"""

from pathlib import Path
import json, random
from datasets import Dataset, DatasetDict
from build_dataset import build_dataset      
from morph_cot  import build_morph_cot    

RAW_PATH = Path("raw_puzzles.jsonl")
print("About to build the dataset")
build_dataset(
    examples=20_000,   
    outfile=RAW_PATH,
    min_words=3,
    max_words=5,
    dict_words=10_000,
    seed=1,
)

SYSTEM_TXT = (
    """You are a word-formation agent for a Bananagrams-style game. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.""" 
)

def make_example(line: str):
    rec  = json.loads(line)
    rack = rec["puzzle"].upper()
    words = rec["solution"]     
    user_prompt = (
        f"Available letters (each can be used once): {' '.join(rack)}\n\n"
        'Form as many valid English words as you can using these letters. '
        'Return your answer as a JSON array of uppercase words, e.g.: ["CAT", "BAT"].'
    )

    cot = build_morph_cot(
        puzzle=rack,
        solution=words,
        unused=len(set(rack)) - len("".join(words)), 
        zipf_scores=rec.get("zipf_scores"),
    )

    assistant_msg = f"<think>\n{cot}\n</think>\n<answer>{json.dumps(words)}</answer>"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_TXT},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_msg},
        ]
    }

examples = [make_example(l) for l in RAW_PATH.read_text().splitlines()]

random.seed(27)
random.shuffle(examples)

# 5 % validation
split = int(len(examples) * 0.05)    
ds = DatasetDict({
    "train": Dataset.from_list(examples[split:]),
    "validation": Dataset.from_list(examples[:split]),
})

LOCAL_DIR = Path("word_puzzle_cot")
ds.save_to_disk(LOCAL_DIR)       

print("Finished!")
