import os
import glob
import csv
import torch
import numpy as np

from typing import Literal
from torch.utils.data import Dataset
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from datasets import load_dataset, Dataset as HFDataset, config as hf_datasets_config

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
    task_name: Literal[
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
    data_path: str
    random_state: int = 42
    dataset_name: str = "InstaDeepAI/nucleotide_transformer_downstream_tasks_revised"


def build_data_splits(config: DatasetConfig):
    base = os.path.join(config.data_path, config.task_name)
    X_train, y_train = read_csv_split(os.path.join(base, "train.csv"))
    X_val, y_val = read_csv_split(os.path.join(base, "dev.csv"))
    X_test, y_test = read_csv_split(os.path.join(base, "test.csv"))
    return X_train, y_train, X_val, y_val, X_test, y_test


def _load_full_split_datasets(dataset_name: str):
    """Return the full (all-task) ``train`` / ``test`` splits of ``dataset_name``.

    datasets==4.2.0 removed ``trust_remote_code`` / loading-script support, so the original
    ``load_dataset(dataset_name, split=..., trust_remote_code=True)`` fails for this script-based
    dataset. When a materialized HF arrow cache exists locally (HF_DATASETS_CACHE), we read the
    prepared per-split arrow files directly -- same rows in the same order as ``load_dataset`` would
    return, so the downstream task filter + stratified split are byte-identical, with no network and
    no ``trust_remote_code``. Falls back to the original loading-script call when no cache is found
    (so non-cached environments still work).
    """
    ns = dataset_name.replace("/", "___")            # HF namespaced cache dir name
    base = dataset_name.split("/")[-1]               # prepared split arrow file prefix
    cache = hf_datasets_config.HF_DATASETS_CACHE     # honors HF_HOME / HF_DATASETS_CACHE env

    def _find(split):
        hits = sorted(glob.glob(os.path.join(cache, ns, "*", "*", "*", f"{base}-{split}.arrow")))
        return hits[0] if hits else None

    train_arrow, test_arrow = _find("train"), _find("test")
    if train_arrow and test_arrow:
        return HFDataset.from_file(train_arrow), HFDataset.from_file(test_arrow)
    return (
        load_dataset(dataset_name, split="train", trust_remote_code=True),
        load_dataset(dataset_name, split="test", trust_remote_code=True),
    )


def build_data_splits_from_huggingface(config: DatasetConfig):
    # --- Load data from HuggingFace ---
    print("Loading dataset from HuggingFace...")
    # datasets==4.2.0 fix: read the local arrow cache directly (no trust_remote_code / network).
    ds_all_train, ds_all_test = _load_full_split_datasets(config.dataset_name)

    # Filter to just this task
    ds_train_full = ds_all_train.filter(lambda ex: ex["task"] == config.task_name)
    ds_test = ds_all_test.filter(lambda ex: ex["task"] == config.task_name)

    # Stratified split for validation
    if len(ds_train_full) > 1 and len(np.unique(ds_train_full["label"])) > 1:
        train_idx, val_idx = train_test_split(
            range(len(ds_train_full)),
            test_size=0.1,
            random_state=config.random_state,
            stratify=ds_train_full["label"],
        )
    else:
        train_idx, val_idx = train_test_split(
            range(len(ds_train_full)), test_size=0.1, random_state=config.random_state
        )

    ds_train = ds_train_full.select(train_idx)
    ds_val = ds_train_full.select(val_idx)

    # Extract sequences and labels
    X_train, y_train = ds_train["sequence"], ds_train["label"]
    X_val, y_val = ds_val["sequence"], ds_val["label"]
    X_test, y_test = ds_test["sequence"], ds_test["label"]

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    return X_train, y_train, X_val, y_val, X_test, y_test


class SeqDataset(Dataset):
    def __init__(self, sequences, labels, max_len, teacher_logits=None, teacher_features=None):
        self.ids = torch.tensor([encode_seq(s, max_len) for s in sequences], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_logits = (
            torch.tensor(teacher_logits, dtype=torch.float) if teacher_logits is not None else None
        )
        self.teacher_features = (
            torch.tensor(teacher_features, dtype=torch.float)
            if teacher_features is not None
            else None
        )
        assert self.teacher_logits is None or len(self.teacher_logits) == len(self.ids)
        assert self.teacher_features is None or len(self.teacher_features) == len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        x = self.ids[idx]
        y = self.labels[idx]
        if self.teacher_logits is not None and self.teacher_features is not None:
            return x, y, self.teacher_logits[idx], self.teacher_features[idx]
        elif self.teacher_logits is not None:
            return x, y, self.teacher_logits[idx]
        return x, y


class EmbeddingSeqDataset(Dataset):
    """SeqDataset variant for the R1.3 embedding-input student (input_mode='nt_embedding').

    Identical batch contract to ``SeqDataset`` EXCEPT batch element 0 is the per-bp teacher
    embedding ``[max_len, hidden_size]`` (float) instead of the char-index ids ``[max_len]``
    (long). The student's ``BPNetClassifier.forward`` consumes it via its learned input adapter;
    labels and the (pooled) teacher logits/features KD targets are unchanged, so the trainer,
    losses and best-ckpt selection are byte-identical to the one-hot path.

    ``embeddings`` is the [N, max_len, hidden_size] array from
    ``embedding_cache.precompute_perbp_embeddings`` (may be a numpy memmap; copied to a tensor
    lazily per item to avoid materializing the whole array on GPU).

    TRUE one-hot late-fusion (front_end='latefuse_onehot'): pass ``sequences`` + ``max_len`` so the
    REAL one-hot ids are computed (exactly like the baseline ``SeqDataset``) and APPENDED to the
    embedding along the feature axis -> element 0 is ``[max_len, hidden_size + 4]`` (embedding ++
    one-hot fp). Packing both into a single tensor keeps the trainer's positional batch contract
    (``batch[0]`` is the student input, ``batch[2/3]`` stay logits/features) unchanged -- no trainer
    edits. The student's forward splits ``[..., :D]`` (embedding) and ``[..., D:]`` (one-hot).
    """

    def __init__(self, embeddings, labels, teacher_logits=None, teacher_features=None,
                 sequences=None, max_len=None):
        self.embeddings = embeddings  # np.ndarray [N, L, H] (possibly mmap)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.teacher_logits = (
            torch.tensor(teacher_logits, dtype=torch.float) if teacher_logits is not None else None
        )
        self.teacher_features = (
            torch.tensor(teacher_features, dtype=torch.float)
            if teacher_features is not None
            else None
        )
        # Optional REAL one-hot (latefuse_onehot). Precompute the [N, L, 4] one-hot once (cheap) so
        # __getitem__ just concatenates it onto the embedding. Built from the raw sequences exactly as
        # the baseline SeqDataset does (encode_seq -> A/C/G/T idx -> one_hot).
        self.onehot = None
        if sequences is not None:
            assert max_len is not None, "latefuse_onehot needs max_len to encode the one-hot"
            ids = torch.tensor([encode_seq(s, max_len) for s in sequences], dtype=torch.long)
            self.onehot = torch.nn.functional.one_hot(ids, num_classes=4).float()  # [N, L, 4]
            assert len(self.onehot) == len(self.embeddings)
        assert len(self.embeddings) == len(self.labels)
        assert self.teacher_logits is None or len(self.teacher_logits) == len(self.labels)
        assert self.teacher_features is None or len(self.teacher_features) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(np.ascontiguousarray(self.embeddings[idx])).float()  # [L, D]
        if self.onehot is not None:
            x = torch.cat([x, self.onehot[idx]], dim=-1)  # [L, D+4]: embedding ++ real one-hot
        y = self.labels[idx]
        if self.teacher_logits is not None and self.teacher_features is not None:
            return x, y, self.teacher_logits[idx], self.teacher_features[idx]
        elif self.teacher_logits is not None:
            return x, y, self.teacher_logits[idx]
        return x, y
