#!/usr/bin/env python3
"""
build_word_puzzle_sft_jsonl.py
Create train/validation splits in the ChatML/`messages` format required by
open_r1's SFT script, and save them as JSONL.
"""

import json
import random
from pathlib import Path

from build_dataset import build_dataset    
from morph_cot import build_morph_cot      

RAW_JSONL = Path("raw_puzzles.jsonl")
TRAIN_JSONL = Path("train.jsonl")
VAL_JSONL   = Path("validation.jsonl")

N_EXAMPLES     = 20_000   # total examples to generate
DICT_WORDS     = 10_000   # top‑N words from wordfreq
MIN_WORDS      = 3
MAX_WORDS      = 5
SEED_RAW       = 28      # seed for puzzle gen
SEED_SPLIT     = 45      # seed for train/val shuffle
VAL_FRACTION   = 0.05     # 5% validation

SYSTEM_TXT = (
    "You are a word-formation agent for a Bananagrams-style game. "
    "A conversation between User and Assistant. The user asks a question, "
    "and the Assistant solves it. The assistant first thinks about the reasoning "
    "process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags."
)

print(f"[1/4] Generating {N_EXAMPLES} raw puzzles → {RAW_JSONL}")
build_dataset(
    examples=N_EXAMPLES,
    outfile=RAW_JSONL,
    min_words=MIN_WORDS,
    max_words=MAX_WORDS,
    dict_words=DICT_WORDS,
    seed=SEED_RAW,
)

def make_example(record: dict) -> dict:
    rack   = record["puzzle"].upper()
    words  = record["solution"]
    # crude leftover count
    unused = len(rack) - sum(len(w) for w in words)

    user_prompt = (
        f"Available letters (each can be used once): {' '.join(rack)}\n\n"
        "Form as many valid English words as you can using these letters. "
        'Return your answer as a JSON array of uppercase words, e.g.: ["CAT", "BAT"].'
    )

    cot = build_morph_cot(
        puzzle=rack,
        solution=words,
        unused=unused,
        zipf_scores=record.get("zipf_scores"),
    )

    assistant_msg = (
        "<think>\n" + cot + "\n</think>\n"
        "<answer>" + json.dumps(words) + "</answer>"
    )

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_TXT},
            {"role": "user",      "content": user_prompt},
            {"role": "assistant", "content": assistant_msg},
        ]
    }

print("[2/4] Transforming raw puzzles → ChatML examples")
raw_lines = RAW_JSONL.read_text().splitlines()
examples = [ make_example(json.loads(line)) for line in raw_lines ]

print("[3/4] Shuffling and splitting into train/validation")
random.seed(SEED_SPLIT)
random.shuffle(examples)
n_val = int(len(examples) * VAL_FRACTION)

val_set   = examples[:n_val]
train_set = examples[n_val:]

print(f"[4/4] Writing {len(train_set)} train → {TRAIN_JSONL}")
with TRAIN_JSONL.open("w", encoding="utf-8") as fh:
    for ex in train_set:
        fh.write(json.dumps(ex) + "\n")

print(f"[4/4] Writing {len(val_set)} validation → {VAL_JSONL}")
with VAL_JSONL.open("w", encoding="utf-8") as fh:
    for ex in val_set:
        fh.write(json.dumps(ex) + "\n")

print("All done! 🎉")
