"""
Memory buffers for DCFR agents.
"""
from collections import deque
import random

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, info_set, strategy, iteration):
        self.buffer.append((info_set, strategy, iteration))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
