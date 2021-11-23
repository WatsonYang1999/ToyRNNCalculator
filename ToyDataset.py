from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class ToyDataset(Dataset):
    def __init__(self, features: List[str], labels: List[str], seq_lens: List[int]):
        assert len(features) == len(labels) == len(seq_lens)

        self.features = features
        self.labels = labels
        self.seq_lens = seq_lens



    def __len__(self):
        return len(self.features)

    def __getitem__(self, item):
        feature = self.features[item]
        label = self.labels[item]
        feature = [char2index(i) for i in feature]
        label = [char2index(i) for i in label]
        seq_len = self.seq_lens[item]

        feature = np.array(feature, dtype=np.int32)
        label = np.array(label, dtype=np.int32)


        return torch.LongTensor(feature), torch.LongTensor(label), seq_len


def char2index(ch):
    if ch == '$':
        x = 0  # pad
    elif ch == '#':
        x = 1  # EOS
    elif ch == '@':
        x = 2  # BOS
    elif ch == '+':
        x = 3
    elif ch == '-':
        x = 4
    elif ch == '*':
        x = 5
    elif ch == '/':
        x = 6
    elif ch == '(':
        x = 7
    elif ch == ')':
        x = 8
    elif ch == '.':
        x = 9
    elif ch == 'e':
        x = 10
    else:

        x = ord(ch) - ord('0') + 11
        if x >= 21 :
            print(ch)
    return x


class PadSequence(object):
    def __call__(self, batch: List[Tuple[torch.Tensor]]):

        features_padded = torch.nn.utils.rnn.pad_sequence([x[0] for x in batch], batch_first=True)
        labels_padded = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True)
        seq_lens = []
        for x in batch:
            seq_lens.append(x[2])

        seq_lens = torch.LongTensor(seq_lens)

        return features_padded, labels_padded, seq_lens
