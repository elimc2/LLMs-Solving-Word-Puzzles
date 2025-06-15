#!/usr/bin/env python3
"""
word_puzzle_dataset.py
======================

Generate a JSON‑Lines (JSONL) dataset for a word‑formation puzzle game.

Key flag
--dict_words N   Keep only the N most‑frequent English words (via wordfreq).

Example:
python word_puzzle_dataset.py --examples 100 --dict_words 5000 --seed 42
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from wordfreq import iter_wordlist, zipf_frequency  # type: ignore
except ImportError:
    sys.exit("ERROR: `wordfreq` missing.  Install with `pip install wordfreq`")

# The three ranges divide the English vocabulary (as provided by the wordfreq library) into tiers of how often a word appears in real-world text
BUCKETS: Tuple[Tuple[float, float], ...] = (
    (5.5, 99.0),  # common
    (4.0, 5.5),   # mid
    (0.0, 4.0),   # rare
)
BUCKET_NAMES: Tuple[str, ...] = ("common", "mid", "rare")

DEFAULT_LEN_PROBS: Dict[int, float] = {
    2: 0.08, 3: 0.12, 4: 0.18, 5: 0.18,
    6: 0.16, 7: 0.14, 8: 0.10, 9: 0.04,
}

RNG = random.Random()

def shuffle_letters(words: Sequence[str]) -> str:
    """Return a random permutation of all letters in *words*."""
    letters = [c for w in words for c in w]
    RNG.shuffle(letters)
    return "".join(letters)


def load_word_buckets(max_dict_words: int | None = None) -> Dict[str, List[str]]:
    """
    Load english words from *wordfreq*, partitioned into Zipf buckets.
    If *max_dict_words* is given keep only that many most‑frequent tokens.
    """
    buckets: Dict[str, List[str]] = defaultdict(list)
    words_iter = iter_wordlist("en", wordlist="best")
    if max_dict_words is not None:
        words_iter = itertools.islice(words_iter, max_dict_words)

    for w in words_iter:
        if not w.isalpha():
            continue
        z = zipf_frequency(w, "en")
        for (low, high), name in zip(BUCKETS, BUCKET_NAMES):
            if low <= z < high:
                buckets[name].append(w.upper())
                break
    return buckets


def choose_word(
    candidates: List[str],
    length_probs: Dict[int, float],
    fallback: List[str],
) -> str:
    """Pick a word; if *candidates* empty, pick from *fallback* instead."""
    if not candidates:
        candidates = fallback

    by_len: Dict[int, List[str]] = defaultdict(list)
    for w in candidates:
        by_len[len(w)].append(w)

    pairs = [
        (L, length_probs.get(L, 0.0))
        for L in by_len
        if length_probs.get(L, 0.0) > 0
    ]

    if not pairs:  # no weights → uniform
        return RNG.choice(candidates)

    lengths, weights = zip(*pairs)
    chosen_len = RNG.choices(lengths, weights)[0]
    return RNG.choice(by_len[chosen_len])


def make_example(
    word_buckets: Dict[str, List[str]],
    num_words: int,
    length_probs: Dict[int, float],
) -> Dict:
    """Create one puzzle record."""
    buckets_seq = ["common"] + RNG.choices(["mid", "rare"], k=num_words - 1)

    words, zipfs = [], []
    for bucket in buckets_seq:
        w = choose_word(
            word_buckets[bucket],
            length_probs,
            fallback=word_buckets["common"],
        )
        words.append(w)
        zipfs.append(zipf_frequency(w.lower(), "en"))

    return {
        "puzzle": shuffle_letters(words),
        "solution": words,
        "zipf_scores": zipfs,
    }


def build_dataset(
    examples: int,
    outfile: Path,
    min_words: int,
    max_words: int,
    dict_words: int | None,
    length_probs: Dict[int, float] = DEFAULT_LEN_PROBS,
    seed: int | None = None,
) -> None:
    """Generate *examples* puzzles and save to *outfile* (JSONL)."""
    if seed is not None:
        RNG.seed(seed)

    buckets = load_word_buckets(dict_words)

    with outfile.open("w", encoding="utf-8") as fh:
        for _ in range(examples):
            n = RNG.randint(min_words, max_words)
            fh.write(json.dumps(make_example(buckets, n, length_probs)) + "\n")


# CLI                                                                    
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create word‑puzzle JSONL dataset")
    p.add_argument("--examples", type=int, default=1000,
                   help="Number of puzzles to generate (default 1000)")
    p.add_argument("--outfile", type=Path, default=Path("puzzles.jsonl"),
                   help="Output JSONL path (default puzzles.jsonl)")
    p.add_argument("--min_words", type=int, default=3,
                   help="Minimum words per puzzle (default 3)")
    p.add_argument("--max_words", type=int, default=5,
                   help="Maximum words per puzzle (default 5)")
    p.add_argument("--dict_words", type=int,
                   help="Limit dictionary to N most frequent words")
    p.add_argument("--seed", type=int, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(
        examples=args.examples,
        outfile=args.outfile,
        min_words=args.min_words,
        max_words=args.max_words,
        dict_words=args.dict_words,
        seed=args.seed,
    )
    print(f"Wrote {args.examples} puzzles → {args.outfile}")


if __name__ == "__main__":
    main()
