"""
Specialized memory buffer for LSTM-based DCFR agents.
"""
from collections import deque
import random

class LSTMMemory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, private_card, public_card, history, strategy, iteration):
        # Store components separately
        self.buffer.append((private_card, public_card, history, strategy, iteration))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
