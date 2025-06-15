import argparse
import logging
import time
from collections import Counter
from typing import List, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Optional third‑party resources
# ──────────────────────────────────────────────────────────────────────────────
try:                                         # 1️⃣ english‑words
    from english_words import get_english_words_set

    _USE_ENGLISH_WORDS = True
except ImportError:                          # noqa: E402 – allow fallback import
    _USE_ENGLISH_WORDS = False

try:                                         # 2️⃣ wordfreq
    from wordfreq import top_n_list

    _USE_WORDFREQ = True
except ImportError:
    _USE_WORDFREQ = False

try:                                         # 3️⃣ PyEnchant for spell‑check
    import enchant

    _ENCHANT_DICT = enchant.Dict("en_US")
except (ImportError, AttributeError, enchant.errors.DictNotFoundError):
    _ENCHANT_DICT = None

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MIN_WORD_LEN = 2           # ignore 1‑letter words outright
TIMEOUT_SEC   = 15         # default DFS wall clock limit

# Scrabble‑legal 2‑letter words (North‑American list, no abbreviations)
ALLOWED_TWO_LETTERS = {
    "AA","AB","AD","AE","AG","AH","AI","AL","AM","AN","AR","AS","AT","AW","AX",
    "AY","BA","BE","BI","BO","BY","DA","DE","DO","ED","EF","EH","EL","EM","EN",
    "ER","ES","ET","EW","EX","FA","FE","GO","GU","HA","HE","HI","HM","HO","ID",
    "IF","IN","IS","IT","JA","JO","KA","KI","LA","LI","LO","MA","ME","MI","MM",
    "MO","MU","MY","NA","NE","NO","NU","OD","OE","OF","OH","OI","OK","OM","ON",
    "OO","OP","OR","OS","OU","OW","OX","OY","PA","PE","PI","PO","QI","RE","SH",
    "SI","SO","TA","TE","TI","TO","UH","UM","UN","UP","UR","US","UT","WE","WO",
    "XI","XU","YA","YE","YO","YU","ZA"
}

# ──────────────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────────────
_LOG = logging.getLogger("letter_solver")


def _configure_logging(verbose: bool = False) -> None:
    """Set up root logger with a sensible format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Dictionary loading / validation
# ──────────────────────────────────────────────────────────────────────────────
def load_dictionary(
    path: str = "/usr/share/dict/words",
    freq_limit: int = 200_000,
) -> List[str]:
    """
    Return a list of **uppercase** words.

    Tries, in order:
    1. ``english‑words`` (`gcide` + `web2` sources).
    2. ``wordfreq`` top‑*freq_limit* words.
    3. A plain text fallback file *path*.

    Parameters
    ----------
    path
        Location of a newline‑delimited text file (one word per line) used
        only if neither package is available.
    freq_limit
        If *wordfreq* is used, how many of the most frequent tokens to keep.

    Raises
    ------
    SystemExit
        If no dictionary source is available.
    """
    t0 = time.perf_counter()

    if _USE_ENGLISH_WORDS:
        _LOG.debug("Loading dictionary via english‑words …")
        words = [
            w.upper()
            for w in get_english_words_set(["gcide", "web2"], lower=False)
            if w.isalpha()
        ]

    elif _USE_WORDFREQ:
        _LOG.debug("Loading dictionary via wordfreq (top_n_list, n=%d) …", freq_limit)
        words = [w.upper() for w in top_n_list("en", freq_limit) if w.isalpha()]

    else:
        _LOG.debug("Loading dictionary from file %s …", path)
        try:
            with open(path, encoding="utf-8") as fh:
                words = [ln.strip().upper() for ln in fh if ln.strip().isalpha()]
        except FileNotFoundError as exc:
            msg = (
                f"ERROR: No dictionary available. Searched english‑words, wordfreq, "
                f"and file at {path}."
            )
            _LOG.critical(msg)
            raise SystemExit(msg) from exc

    _LOG.info("Loaded %d words (%.1f s)", len(words), time.perf_counter() - t0)
    return words


def is_valid_word(word: str) -> bool:
    """
    Return ``True`` if *word* passes the optional spell‑checker.

    If *pyenchant* is present, ``en_US`` is used; otherwise everything is
    accepted.  Called only on candidate words already in our base list, so
    it mainly filters aggressive frequency lists.
    """
    return _ENCHANT_DICT.check(word) if _ENCHANT_DICT else True


# ─── DICTIONARY FILTER ───────────────────────────────────────────────────────
def filter_words(words: Sequence[str], letter_counts: Counter) -> List[str]:
    """
    Keep only buildable words; apply MIN_WORD_LEN & 2‑letter whitelist.
    """
    can_build = []
    for w in words:
        if len(w) < MIN_WORD_LEN:
            continue
        if len(w) == 2 and w not in ALLOWED_TWO_LETTERS:
            continue

        wc = Counter(w)
        if all(wc[ch] <= letter_counts.get(ch, 0) for ch in wc) and is_valid_word(w):
            can_build.append(w)

    _LOG.info(f"Candidate words after filtering: {len(can_build):,d}")
    return can_build

# ─── GREEDY SEED & TIMEOUT IN DFS ────────────────────────────────────────────
import time

# ─── GREEDY SEED & TIMEOUT IN DFS ────────────────────────────────────────────
def find_best_cover(
    words: Sequence[str],
    letter_counts: Counter,
    timeout: float = TIMEOUT_SEC,
) -> Tuple[List[str], int]:
    """
    Branch‑and‑bound DFS with:
      • greedy initial solution (longest single buildable word);
      • wall‑clock timeout.

    Returns
    -------
    best_subset : list[str]
        Words covering the maximum number of letters found so far.
    used : int
        Total letters used by *best_subset*.
    """
    deadline = time.time() + timeout
    words = sorted(words, key=len, reverse=True)
    n = len(words)

    # ── Greedy seed ──────────────────────────────────────────────────────────
    best: dict[str, object] = {"used": 0, "subset": []}  # record of best so far
    chosen0: List[str] = []                              # initial path for DFS
    for w in words:
        wc = Counter(w)
        if all(c <= letter_counts[ch] for ch, c in wc.items()):
            best["subset"] = [w]
            best["used"] = len(w)
            chosen0 = [w]                    # put the word on the search stack
            for ch, c in wc.items():         # remove its letters from the pool
                letter_counts[ch] -= c
            break

    # suffix length sum for branch‑and‑bound
    rem_len = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        rem_len[i] = rem_len[i + 1] + len(words[i])

    # ── Depth‑first search ───────────────────────────────────────────────────
    def dfs(idx: int, counts: Counter, chosen: List[str], used: int) -> None:
        if time.time() > deadline:
            return
        if used + rem_len[idx] <= best["used"]:
            return
        if used > best["used"]:
            best["used"] = used
            best["subset"] = chosen.copy()
            _LOG.debug(f"New best {best['used']} letters: {best['subset']}")

        for i in range(idx, n):
            w = words[i]
            if len(w) > sum(counts.values()):
                continue
            wc = Counter(w)
            if all(wc[ch] <= counts.get(ch, 0) for ch in wc):
                new_counts = counts.copy()
                for ch, cnt in wc.items():
                    new_counts[ch] -= cnt
                chosen.append(w)
                dfs(i + 1, new_counts, chosen, used + len(w))
                chosen.pop()

    _LOG.info(f"Starting DFS over {n:,d} words … timeout {timeout}s")
    dfs(0, letter_counts.copy(), chosen0, best["used"])
    return best["subset"], best["used"]


def solve(
    letters: str,
    dictionary_path: str | None = None,
    freq_limit: int = 200_000,
) -> Tuple[List[str], int]:
    """
    Solve for the *letters* string.

    Parameters
    ----------
    letters
        The pool of letters available (case‑insensitive).
    dictionary_path
        Optional local word‑file override.
    freq_limit
        If *wordfreq* is the dictionary source, how many top words to load.

    Returns
    -------
    best_subset
        List of words covering the maximum possible letters.
    unused
        Number of letters left unused.
    """
    letters = letters.upper()
    letter_counts = Counter(letters)
    _LOG.info("Letter pool: %s (%d letters)", letters, len(letters))

    # Choose appropriate dictionary loader
    words = (
        load_dictionary(dictionary_path or "/usr/share/dict/words", freq_limit)
        if dictionary_path or (not _USE_ENGLISH_WORDS and not _USE_WORDFREQ)
        else load_dictionary(freq_limit=freq_limit)
    )

    candidates = filter_words(words, letter_counts)
    best_subset, used = find_best_cover(candidates, letter_counts)
    unused = len(letters) - used
    return best_subset, unused


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Command‑line interface definition."""
    parser = argparse.ArgumentParser(
        description="Find the maximal set of words using the supplied letters."
    )
    parser.add_argument("letters", help="Pool of letters (no separators).")
    parser.add_argument(
        "--dict",
        dest="dict_path",
        metavar="PATH",
        help="Override dictionary text file path.",
    )
    parser.add_argument(
        "--freq",
        dest="freq_limit",
        type=int,
        default=200_000,
        metavar="N",
        help="If using wordfreq, load the top‑N most frequent words (default 200 000).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG‑level logging for tracing.",
    )
    # ─── CLI: add --timeout flag (optional) ──────────────────────────────────────
    parser.add_argument(
        "--timeout",
        type=float,
        default=TIMEOUT_SEC,
        metavar="SEC",
        help=f"Abort search after SEC seconds (default {TIMEOUT_SEC}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """CLI wrapper."""
    args = _parse_args(argv)
    _configure_logging(args.verbose)

    best_set, unused = solve(args.letters, args.dict_path, args.freq_limit)

    print("Best word set:", best_set)
    print(f"Letters unused: {unused}")


if __name__ == "__main__":
    main()