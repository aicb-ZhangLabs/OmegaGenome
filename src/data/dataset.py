import os
import csv
import torch

from typing import List, Literal
from torch.utils.data import Dataset
from dataclasses import dataclass

CHAR2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

# special tasks with more than 2 labels, if not specified, default to 2
TASK2LABELNUM = {"enhancers_types": 3, "splice_sites_all": 3}


def encode_seq(seq: str, max_len: int):
    seq = seq.upper()
    ids = [CHAR2IDX.get(ch, 0) for ch in seq[:max_len]]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


def get_num_labels(task_name):
    return TASK2LABELNUM.get(task_name, 2)


def read_csv_split(split_csv):
    with open(split_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader)
        seqs, labs = zip(*[(r[0], int(r[1])) for r in reader])
    return list(seqs), list(labs)


@dataclass
class DatasetConfig:
    task_name: List[
        Literal[
            "H2AFZ",
            "H3K27ac",
            "H3K27me3",
            "H3K36me3",
            "H3K4me1",
            "H3K4me2",
            "H3K4me3",
            "H3K9ac",
            "H3K9me3",
            "H4K20me1",
            "promoter_all",
            "promoter_tata",
            "promoter_no_tata",
            "enhancers",
            "enhancers_types",
            "splice_sites_all",
            "splice_sites_acceptors",
            "splice_sites_donors",
        ]
    ]
    data_path: str


def build_data_splits(config: DatasetConfig):
    base = os.path.join(config.data_path, config.task_name)
    X_train, y_train = read_csv_split(os.path.join(base, "train.csv"))
    X_val, y_val = read_csv_split(os.path.join(base, "dev.csv"))
    X_test, y_test = read_csv_split(os.path.join(base, "test.csv"))
    return X_train, y_train, X_val, y_val, X_test, y_test


class SeqDataset(Dataset):
    def __init__(self, sequences, labels, max_len, teacher_logits=None):
        self.ids = torch.tensor(
            [encode_seq(s, max_len) for s in sequences], dtype=torch.long
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_logits = (
            torch.tensor(teacher_logits, dtype=torch.float)
            if teacher_logits is not None
            else None
        )
        assert self.teacher_logits is None or len(self.teacher_logits) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        y = self.labels[idx]
        if self.teacher_logits is not None:
            return x, y, self.teacher_logits[idx]
        return x, y
