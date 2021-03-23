import torch
import torch.nn as nn
import numpy as np


class UniformNegativeSampler(nn.Module):
    def __init__(self, n_items, exclude_pos=False):
        nn.Module.__init__(self)
        # number of items, including the padding item 0
        self.n_items = n_items
        self.exclude_pos = exclude_pos

    def forward(self, k, pos_targets):
        if self.exclude_pos:
            candidate_set = list(set(range(1, self.n_items)) - set(pos_targets))
            neg_targets = np.random.choice(candidate_set, size=(len(pos_targets), k))
            neg_targets = torch.tensor(neg_targets, dtype=torch.int32)
            return neg_targets
        else:
            return torch.randint(1, self.n_items, [len(pos_targets), k])
