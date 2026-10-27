import hashlib
import re

import numpy as np

PROJECT = "/work/test-project"


class FakeEmbedder:
    def __init__(self, dimensions=32):
        self.dimensions = dimensions
        self.calls = []

    def encode(self, texts, *, query=False):
        self.calls.append((texts, query))
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                slot = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4]) % self.dimensions
                vectors[i, slot] += 1
            if not vectors[i].any():
                vectors[i, 0] = 1
        return vectors
