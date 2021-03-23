import numpy as np
from tqdm import tqdm


class Dataset:
    def __init__(self, num_items, user_seq, timestamps=None, idx_user_map=None, idx_item_map=None):
        super().__init__()
        # item 0 is for padding
        self.num_items = num_items
        self.user_seq = user_seq
        self.num_users = len(self.user_seq)
        if timestamps:
            self.timestamps = timestamps
        if idx_item_map:
            self.idx_item_map = idx_item_map
        if idx_user_map:
            self.idx_user_map = idx_user_map

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, idx):
        if hasattr(self, "timestamps"):
            return self.user_seq[idx], self.timestamps[idx]
        else:
            return self.user_seq[idx]


class TrainingDataset(Dataset):
    def __init__(self, num_items, user_seq, timestamps=None, idx_user_map=None, idx_item_map=None, max_seq_len=None):
        super().__init__(num_items, user_seq, timestamps, idx_user_map, idx_item_map)
        if max_seq_len:
            self.max_seq_len = max_seq_len
        self.split()

    def split(self):
        self.train_seq = []
        self.train_targets = []
        if hasattr(self, "timestamps"):
            self.train_seq_timestamps = []
        print("building training dataset...")
        for idx in tqdm(range(self.num_users)):
            seq = self.user_seq[idx]
            if len(seq) < 4:
                continue
            if hasattr(self, "timestamps"):
                seq_timestamps = self.timestamps[idx]
            last_pos = len(seq) - 1
            if hasattr(self, "max_seq_len"):
                for b in range((last_pos + self.max_seq_len - 1) // self.max_seq_len):
                    if (last_pos - b * self.max_seq_len) > self.max_seq_len * 1.1:
                        self.train_targets.append(
                            seq[(last_pos - (b + 1) * self.max_seq_len):(last_pos - b * self.max_seq_len)])
                        self.train_seq.append(
                            seq[(last_pos - (b + 1) * self.max_seq_len - 1):(last_pos - b * self.max_seq_len - 1)])
                        if hasattr(self, "timestamps"):
                            self.train_seq_timestamps.append(seq_timestamps[(
                                                                                    last_pos - (
                                                                                        b + 1) * self.max_seq_len - 1):(
                                                                                        last_pos - b * self.max_seq_len - 1)])
                    else:
                        self.train_targets.append(
                            seq[1:(last_pos - b * self.max_seq_len)])
                        self.train_seq.append(
                            seq[0:(last_pos - b * self.max_seq_len - 1)])
                        if hasattr(self, "timestamps"):
                            self.train_seq_timestamps.append(
                                seq_timestamps[0:(last_pos - b * self.max_seq_len - 1)])
                        break
            else:
                self.train_targets.append(seq[1:last_pos])
                self.train_seq.append(seq[0:last_pos - 1])
                if hasattr(self, "timestamps"):
                    self.train_seq_timestamps.append(
                        seq_timestamps[0:last_pos - 1])

    def __len__(self):
        return len(self.train_seq)

    def __getitem__(self, idx):
        if hasattr(self, "timestamps"):
            return self.train_seq[idx], self.train_targets[idx], self.train_seq_timestamps[idx]
        else:
            return self.train_seq[idx], self.train_targets[idx]


class EvaluationDataset(Dataset):
    def __init__(self, num_items, user_seq, timestamps=None, idx_user_map=None, idx_item_map=None, max_seq_len=None,
                 num_negatives=100):
        super().__init__(num_items, user_seq, timestamps, idx_user_map, idx_item_map)
        if max_seq_len:
            self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.split()

    def split(self):
        self.eval_seq = []
        self.eval_targets = []
        self.eval_neg_samples = []
        if hasattr(self, "timestamps"):
            self.eval_seq_timestamps = []
        print("building evaluation dataset...")
        for idx in tqdm(range(self.num_users)):
            seq = self.user_seq[idx]
            if len(seq) < 4:
                continue
            last_pos = len(seq) - 1
            self.eval_targets.append(seq[last_pos:last_pos + 1])
            neg_candidate_set = list(set(range(1, self.num_items)) - set(seq))
            neg_targets = np.random.choice(
                neg_candidate_set, size=self.num_negatives)
            self.eval_neg_samples.append(neg_targets)
            if hasattr(self, "timestamps"):
                seq_timestamps = self.timestamps[idx]
            if hasattr(self, "max_seq_len"):
                self.eval_seq.append(
                    seq[max(0, last_pos - self.max_seq_len):last_pos])
                if hasattr(self, "timestamps"):
                    self.eval_seq_timestamps.append(
                        seq_timestamps[max(0, last_pos - self.max_seq_len):last_pos])
            else:
                self.eval_seq.append(seq[0:last_pos])
                if hasattr(self, "timestamps"):
                    self.eval_seq_timestamps.append(seq_timestamps[0:last_pos])

    def __len__(self):
        return len(self.eval_seq)

    def __getitem__(self, idx):
        if hasattr(self, "timestamps"):
            return self.eval_seq[idx], self.eval_targets[idx], self.eval_seq_timestamps[idx], self.eval_neg_samples[idx]
        else:
            return self.eval_seq[idx], self.eval_targets[idx], self.eval_neg_samples[idx]
