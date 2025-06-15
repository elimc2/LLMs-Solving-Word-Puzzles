#!/usr/bin/env python3
"""
morph_cot_v2.py
Refined Chain-of-Thought (CoT) generator for word-formation puzzles.
This version builds each word only after listing all morphological clues,
adds fallback reasoning, and handles small words specially.
"""

from collections import Counter
from typing import List, Sequence, Optional
try:
    from wordfreq import zipf_frequency  # type: ignore
except ImportError:
    zipf_frequency = None

# Tunable morphological lists
COMMON_PREFIXES = (
    "RE", "UN", "IN", "DIS", "EN", "EM", "NON", "NO", "CON", "OVER", "MIS", "SUB",
    "PRE", "INTER", "FORE", "DE", "TRANS", "SUPER", "SEM", "ANTI",
)

COMMON_SUFFIXES = (
    "S", "ES", "ED", "ING", "LY", "ER", "OR", "ION", "TION", "AL",
    "NESS", "MENT", "FUL", "LESS", "ABLE", "IBLE", "SELF",
)

RARE_LETTERS = set("JQXZK")
## Add common consonant clusters (digraphs)
CLUSTERS = ("TH", "SH", "CH", "PH", "WH", "CK", "QU", "WR")

def pool_to_markdown(pool: Counter[str], highlight: Sequence[str] | None = None) -> str:
    """Return Markdown with highlight letters struck through."""
    counts = pool.copy()
    strikes = Counter(highlight or [])
    pieces: List[str] = []
    for ch in sorted(counts):
        for _ in range(counts[ch]):
            if strikes[ch]:
                pieces.append(f"~~{ch}~~")
                strikes[ch] -= 1
            else:
                pieces.append(ch)
    return " ".join(pieces) or "-- none --"

def explain_choice(word: str) -> str:
    """Fallback morphology-based explanation."""
    parts: List[str] = []
    for pre in COMMON_PREFIXES:
        if word.startswith(pre):
            parts.append(f"begins with the prefix {pre}-")
            break
    for suf in COMMON_SUFFIXES:
        if word.endswith(suf):
            parts.append(f"ends with the suffix -{suf}")
            break
    rare = set(word) & RARE_LETTERS
    if rare:
        letters = ", ".join(sorted(rare))
        if len(rare) == 1:
            parts.append(f"uses the rare letter {letters}")
        else:
            parts.append(f"uses the rare letters {letters}")
    if not parts:
        parts.append("is the longest or clearest word I see")
    return "It " + "; ".join(parts)

def build_morph_cot(puzzle: str,
                       solution: Sequence[str],
                       unused: int,
                       zipf_scores: Optional[Sequence[float]] = None) -> str:
    """Generate a step-by-step CoT, listing clues before naming each word."""
    puzzle = puzzle.upper()
    pool = Counter(puzzle)
    lines: List[str] = []
    lines.append(f"### Rack: `{puzzle}`")
    lines.append("---")

    VOWELS = set("AEIOU")

    for idx, word in enumerate(solution, start=1):
        w = word.upper()
        lines.append(f"\n**Step {idx}:**")

        # 1) Alphabetize rack
        sorted_letters = " ".join(sorted(pool.elements()))
        lines.append(f"- Alphabetize rack: {sorted_letters}")

        # Collect clues
        clues_found = []

        # 2) Small words (len<=2)
        if len(w) <= 2:
            letters = ", ".join(list(w))
            lines.append(f"- I see the letters {letters} -> that spells the word '{w}'")
            lines.append(f"- '{w}' is a common word")
        else:
            # 3) Suffix
            suffix = next((s for s in COMMON_SUFFIXES if w.endswith(s)
                           and all(pool[c] > 0 for c in s)), None)
            if suffix:
                letters = ", ".join(suffix)
                lines.append(f"- I see the letters {letters} -> suffix '-{suffix}'")
                clues_found.append('suffix')

            # 4) Prefix
            prefix = next((p for p in COMMON_PREFIXES if w.startswith(p)
                           and all(pool[c] > 0 for c in p)), None)
            if prefix:
                letters = ", ".join(prefix)
                lines.append(f"- I see the letters {letters} -> prefix '{prefix}-'")
                clues_found.append('prefix')

            # 5) Vowel pairs
            vowel_pairs = [w[i:i+2] for i in range(len(w)-1)
                           if {w[i], w[i+1]} <= VOWELS
                           and all(pool[c] > 0 for c in w[i:i+2])]
            for vp in vowel_pairs:
                letters = ", ".join(vp)
                lines.append(f"- I also notice the vowel pair '{vp}' ({letters})")
                clues_found.append('vowel')

            # 6) Root/stem breakdown
            if suffix and prefix:
                stem = w[len(prefix):-len(suffix)]
                lines.append(
                    f"- The stem is '{stem}' between prefix '{prefix}-' and suffix '-{suffix}'"
                )
                # Stem validation via Zipf frequency
                if zipf_frequency is not None and stem:
                    z = zipf_frequency(stem.lower(), 'en')
                    if z > 3.0:
                        # lines.append(f"- '{stem}' is also a valid word (Zipf freq {z:.2f})")
                        pass
            elif suffix:
                base = w[:-len(suffix)]
                lines.append(f"- Removing suffix '-{suffix}' leaves base '{base}'")
                # Base validation via Zipf frequency
                if zipf_frequency is not None and base:
                    z = zipf_frequency(base.lower(), 'en')
                    if z > 3.0:
                        # lines.append(f"- '{base}' is also a valid word (Zipf freq {z:.2f})")
                        pass
            elif prefix:
                base = w[len(prefix):]
                lines.append(f"- Removing prefix '{prefix}-' leaves base '{base}'")
                # Base validation via Zipf frequency
                if zipf_frequency is not None and base:
                    z = zipf_frequency(base.lower(), 'en')
                    if z > 3.0:
                        # lines.append(f"- '{base}' is also a valid word (Zipf freq {z:.2f})")
                        pass
            # 6) Consonant-cluster detection
            for cluster in CLUSTERS:
                if cluster in w and all(pool.get(c, 0) > 0 for c in cluster):
                    lines.append(f"- I also notice the consonant cluster '{cluster}'")
                    clues_found.append('cluster')

            # 7) Fallback if no morphological clue
            if not clues_found:
                reason = explain_choice(w)
                lines.append(f"- {reason}.")

        # 7) Name the word
        lines.append(f"- So I form the word '{w}'")

        # 8) Show letters used
        before_md = pool_to_markdown(pool, highlight=list(w))
        lines.append(f"- Rack before using letters: {before_md}")

        # Consume letters
        for ch in w:
            pool[ch] -= 1
            if pool[ch] == 0:
                del pool[ch]

        # 9) Show leftovers
        after_md = pool_to_markdown(pool)
        lines.append(f"- Letters left: {after_md}")

        # 10) Prompt human next step
        lines.append("- What should I do next?")

    # Final summary
    if unused:
        lines.append(f"**Left-over letters:** {pool_to_markdown(pool)}")
    else:
        lines.append("**All letters placed - none left over!**")

    return "\n".join(lines)

if __name__ == '__main__':
    # simple CLI wrap
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('puzzle')
    parser.add_argument('words', nargs='+')
    parser.add_argument('--unused', type=int, default=0)
    args = parser.parse_args()
    print(build_morph_cot(args.puzzle, args.words, args.unused))