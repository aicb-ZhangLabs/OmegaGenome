"""Verify param-matched deployable parameter counts by BUILDING the models and counting.
Run:  PYTHONPATH=. python analysis/verify_param_matched.py
Expected (deployable = with teacher_hidden_size=None, no training-only teacher projection):
  one-hot (original C=64)          = 121,094
  replaceK matched (C=25)          = 120,433  (input_adapter 2560->32 = 81,952 + backbone 38,429 + cls 52)
  latefuse matched (C=34)          = 120,238  (embed_proj 81,952 + fuse_block 6,766 + backbone 31,450 + cls 70)
All within +/-0.6k of the one-hot 121,094 budget => genuinely param-matched.
"""

from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig


def count(label, **kw):
    cfg = BPNetClassifierConfig(num_labels=2, teacher_hidden_size=None, **kw)
    m = BPNetClassifier(cfg)
    total = sum(p.numel() for p in m.parameters())
    bd = {}
    for n, p in m.named_parameters():
        bd[n.split(".")[0]] = bd.get(n.split(".")[0], 0) + p.numel()
    print(f"{label}: DEPLOYABLE TOTAL = {total:,}")
    for k, v in sorted(bd.items(), key=lambda x: -x[1]):
        print(f"    {k:22s} {v:,}")
    return total


if __name__ == "__main__":
    count(
        "one-hot (original C=64) [expect 121,094]",
        input_mode="onehot",
        model_size="original",
        front_end="replace4",
    )
    count(
        "replaceK matched (C=25) [expect 120,433]",
        input_mode="nt_embedding",
        embedding_dim=2560,
        model_size="emb_matched_replaceK",
        front_end="replaceK",
        adapter_width=32,
    )
    count(
        "latefuse matched (C=34) [expect 120,238]",
        input_mode="nt_embedding",
        embedding_dim=2560,
        model_size="emb_matched_latefuse",
        front_end="latefuse_onehot",
        fuse_width=32,
    )
