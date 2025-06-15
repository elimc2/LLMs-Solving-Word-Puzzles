"""
distribution.py

Letter distribution strategies for the Bananagrams-style word game.
"""
import random

class BaseLetterDistribution:
    """Base class for letter distributions."""
    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def sample(self, n):
        """Return a list of n letters sampled according to distribution."""
        raise NotImplementedError

class UniformDistribution(BaseLetterDistribution):
    """Uniformly sample letters from A-Z."""
    def __init__(self, rng=None):
        super().__init__(rng)
        self.letters = [chr(ord('A') + i) for i in range(26)]

    def sample(self, n):
        return [self.rng.choice(self.letters) for _ in range(n)]

class FrequencyDistribution(BaseLetterDistribution):
    """Sample letters according to English letter frequencies."""
    DEFAULT_FREQ = {
        'E': 12.0, 'T': 9.1, 'A': 8.2, 'O': 7.5, 'I': 7.0, 'N': 6.7,
        'S': 6.3, 'R': 6.0, 'H': 5.1, 'L': 4.0, 'D': 3.8, 'C': 3.2,
        'U': 2.8, 'M': 2.4, 'F': 2.2, 'Y': 2.0, 'W': 2.0, 'G': 2.0,
        'P': 1.9, 'B': 1.5, 'V': 1.0, 'K': 0.8, 'X': 0.2, 'Q': 0.1,
        'J': 0.15, 'Z': 0.07
    }

    def __init__(self, freq=None, rng=None):
        super().__init__(rng)
        self.freq = freq or self.DEFAULT_FREQ
        total = sum(self.freq.values())
        self.letters, self.weights = zip(*[(l, w / total) for l, w in self.freq.items()])

    def sample(self, n):
        return self.rng.choices(self.letters, weights=self.weights, k=n)

class ScrabbleDistribution(BaseLetterDistribution):
    """Sample letters according to Scrabble tile distribution."""
    TILES = {
        'A': 9, 'B': 2, 'C': 2, 'D': 4, 'E': 12,
        'F': 2, 'G': 3, 'H': 2, 'I': 9, 'J': 1,
        'K': 1, 'L': 4, 'M': 2, 'N': 6, 'O': 8,
        'P': 2, 'Q': 1, 'R': 6, 'S': 4, 'T': 6,
        'U': 4, 'V': 2, 'W': 2, 'X': 1, 'Y': 2,
        'Z': 1
    }

    def __init__(self, rng=None):
        super().__init__(rng)
        self.bag = []
        for letter, count in self.TILES.items():
            self.bag.extend([letter] * count)

    def sample(self, n):
        if n > len(self.bag):
            raise ValueError(f"Cannot sample {n} letters: bag has only {len(self.bag)} tiles.")
        return self.rng.sample(self.bag, k=n)

class CustomDistribution(BaseLetterDistribution):
    """Sample letters from a user-provided frequency dict."""
    def __init__(self, freq_dict, rng=None):
        super().__init__(rng)
        total = sum(freq_dict.values())
        self.letters, self.weights = zip(*[(l, w / total) for l, w in freq_dict.items()])

    def sample(self, n):
        return self.rng.choices(self.letters, weights=self.weights, k=n)