import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
import numpy as np


def collate_fn(batch, sampler, k=5):
    # src is a list of sequences of tuples (user_id, item_id, timestamp)
    # trg is a list of sequence of postive items
    # batch_users = []
    batch_items = []
    batch_targets = []
    batch_seq_lengths = []
    src, trg = zip(*batch)
    for seq in src:
        batch_items.append(torch.tensor(seq, dtype=torch.long))
        batch_seq_lengths.append(len(seq))
    # (batch_size, L)
    # batch_users = pad_sequence(batch_users, batch_first=True, padding_value=0)
    batch_items = pad_sequence(batch_items, batch_first=True, padding_value=0)
    batch_seq_lengths = torch.tensor(batch_seq_lengths, dtype=torch.long)
    for seq in trg:
        # (len, 1)
        pos_targets = torch.tensor(seq, dtype=torch.long).unsqueeze(1)
        neg_targets = sampler(k, seq)
        # (len, 1 + k)
        targets = torch.cat([pos_targets, neg_targets], dim=-1)
        batch_targets.append(targets)
    # (batch_size, L, 1 + k)
    batch_targets = pad_sequence(
        batch_targets, batch_first=True, padding_value=0)
    # return batch_users, batch_items, batch_targets
    return batch_items, batch_targets, batch_seq_lengths


def collate_fn_with_negatives(batch):
    # src is a list of sequences of tuples (user_id, item_id, timestamp)
    # trg is a list of sequence of postive items
    # batch_users = []
    batch_items = []
    batch_targets = []
    batch_seq_lengths = []
    # negative samples: (batch_size, num_neg)
    src, trg, negative_samples = zip(*batch)
    for seq in src:
        batch_items.append(torch.tensor(seq, dtype=torch.long))
        batch_seq_lengths.append(len(seq))
    # (batch_size, L)
    # batch_users = pad_sequence(batch_users, batch_first=True, padding_value=0)
    batch_items = pad_sequence(batch_items, batch_first=True, padding_value=0)
    batch_seq_lengths = torch.tensor(batch_seq_lengths, dtype=torch.long)
    for seq, neg_samples in zip(trg, negative_samples):
        # len = 1
        # (len, 1)
        pos_targets = torch.tensor(seq, dtype=torch.long).unsqueeze(1)
        # (len, 1 + k)
        targets = torch.cat(
            [pos_targets, torch.tensor(np.expand_dims(neg_samples, axis=0), dtype=torch.long)], dim=-1)
        batch_targets.append(targets)
    # (batch_size, L, 1 + k)
    batch_targets = pad_sequence(
        batch_targets, batch_first=True, padding_value=0)
    # return batch_users, batch_items, batch_targets
    return batch_items, batch_targets, batch_seq_lengths


class LadderSampler(Sampler):
    def __init__(self, data_source, batch_size, fix_order=False):
        super(LadderSampler, self).__init__(data_source)
        # length of all source sequences
        self.seq_lengths = [len(data[0]) for data in data_source]
        self.batch_size = batch_size * 50
        self.fix_order = fix_order

    def __iter__(self):
        if self.fix_order:
            data_samples = zip(self.seq_lengths, np.arange(
                len(self.seq_lengths)), np.arange(len(self.seq_lengths)))
        else:
            data_samples = zip(self.seq_lengths, np.random.permutation(
                len(self.seq_lengths)), np.arange(len(self.seq_lengths)))
        data_samples = sorted(data_samples, key=lambda e: (
            e[1] // self.batch_size, e[0]), reverse=True)
        # e[2] is the index of the data in the original dataset
        return iter(e[2] for e in data_samples)

    def __len__(self):
        return len(self.seq_lengths)
