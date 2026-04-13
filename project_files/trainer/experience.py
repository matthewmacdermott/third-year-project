"""Experience replay buffer and data structures for DQN."""

import random
import numpy as np
from collections import deque
from typing import List, NamedTuple


# Experience tuple for replay buffer
Experience = NamedTuple('Experience', [
    ('state', np.ndarray),
    ('action', int), 
    ('reward', np.ndarray),  # Multi-objective: shape (num_objectives,)
])


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
