"""Fine-tune NTv3 on the NTv3 Benchmark functional tracks (their data); eval on the held-out split.

Reproduces the benchmark teacher: NTv3 backbone + a new T-track head (NTv3FineTune), trained on their
``splits.bed`` train regions with their bigWig targets, evaluated on the test split with the paper's
log(1+x) per-track Pearson. Reuses BenchmarkData, NTv3FineTune, track_distill_loss, per_track_pearson,
and the NTv3 tokenizer.

  python -m src.train.finetune_ntv3 --data_dir <benchmark> --model InstaDeepAI/NTv3_650M_post --out <dir>
"""

import argparse
import json
import os

import numpy as np
import torch

from src.data.ntv3_benchmark import BenchmarkData, load_splits, load_track_meta, sample_windows
from src.model.ntv3_finetune import NTV3_CROP_FRAC, NTv3FineTune
from src.model.ntv3_teacher import NTV3_650M_POST, NTv3Teacher, NTv3TeacherConfig
from src.trainer.track_distill import _teacher_to_btl, track_distill_loss
from src.trainer.track_metrics import per_track_pearson

NTV3_INPUT_MULTIPLE = 128


def _prepare(teacher, bd, coords, window, crop_len, log1p_targets):
    """Tokenize sequences (NTv3) + read central-crop targets aligned to the model's output region."""
    ids = teacher.tokenizer(
        bd.sequences(coords), add_special_tokens=False, padding=True,
        pad_to_multiple_of=NTV3_INPUT_MULTIPLE, return_tensors="pt",
    )["input_ids"]
    off = (window - crop_len) // 2
    crop = [(c, s + off, s + off + crop_len) for (c, s, _e) in coords]
    tgt = torch.from_numpy(bd.targets(crop, nbins=crop_len, log1p=log1p_targets))  # [N, crop_len, T]
    return ids, tgt


@torch.no_grad()
def _evaluate(model, teacher, ids, tgt, device, bs):
    """Per-track Pearson with the paper's log(1+x) on both prediction and target."""
    model.eval()
    preds, tgts = [], []
    for i in range(0, len(ids), bs):
        b = ids[i : i + bs].to(device)
        sp = teacher.model.encode_species(["human"] * b.shape[0]).to(device)
        out = model(b, sp)  # [B, T, L_out]
        preds.append(np.log1p(np.clip(out.float().cpu().numpy(), 0.0, None)))
        tgts.append(np.log1p(np.clip(_teacher_to_btl(tgt[i : i + bs]).numpy(), 0.0, None)))
    return per_track_pearson(np.concatenate(preds), np.concatenate(tgts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="NTv3_benchmark_dataset local dir")
    ap.add_argument("--model", default=NTV3_650M_POST)
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=16384)
    ap.add_argument("--n_train", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--loss", default="poisson", choices=["mse", "poisson", "pearson", "mse+pearson"])
    ap.add_argument("--freeze_backbone", action="store_true", help="head-only (cheap) vs full fine-tune")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits(os.path.join(args.data_dir, "human/splits.bed"))
    meta = load_track_meta(os.path.join(args.data_dir, "benchmark_metadata.tsv"), "human")
    bd = BenchmarkData(os.path.join(args.data_dir, "human/genome.fasta"),
                       os.path.join(args.data_dir, "human/functional_tracks"), meta)
    T = len(meta)
    crop_len = round(args.window * NTV3_CROP_FRAC)
    print(f"tracks={T} window={args.window} crop_len(L_out)={crop_len} device={device} "
          f"mode={'head-only' if args.freeze_backbone else 'full-FT'}")

    teacher = NTv3Teacher(NTv3TeacherConfig(model_name_or_path=args.model, species="human"), device=device)
    model = NTv3FineTune(teacher.model, teacher.model.config.embed_dim, T, args.freeze_backbone).to(device)

    tr_ids, tr_tgt = _prepare(teacher, bd, sample_windows(splits["train"], args.window, n=args.n_train),
                              args.window, crop_len, log1p_targets=False)
    te_ids, te_tgt = _prepare(teacher, bd, sample_windows(splits["test"], args.window, n=args.n_test),
                              args.window, crop_len, log1p_targets=False)
    print(f"train {len(tr_ids)} / test {len(te_ids)} windows")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    best = -1.0
    perm = torch.randperm(len(tr_ids))
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i in range(0, len(tr_ids), args.batch_size):
            idx = perm[i : i + args.batch_size]
            b = tr_ids[idx].to(device)
            sp = teacher.model.encode_species(["human"] * b.shape[0]).to(device)
            opt.zero_grad()
            loss = track_distill_loss(model(b, sp), tr_tgt[idx].to(device), args.loss)
            loss.backward()
            opt.step()
            tot += loss.item()
        mean_r, _ = _evaluate(model, teacher, te_ids, te_tgt, device, args.batch_size)
        print(f"epoch {ep:3d}  train_loss={tot / (len(tr_ids) / args.batch_size):.4f}  test_mean_pearson={mean_r:.4f}")
        if mean_r > best:
            best = mean_r
            torch.save(model.head.state_dict(), os.path.join(args.out, "head_best.pt"))

    mean_r, per = _evaluate(model, teacher, te_ids, te_tgt, device, args.batch_size)
    result = {"model": args.model, "freeze_backbone": args.freeze_backbone, "loss": args.loss,
              "n_tracks": T, "best_test_mean_pearson": best, "final_test_mean_pearson": float(mean_r),
              "per_track_pearson": {meta[i].file_id: (None if np.isnan(per[i]) else float(per[i])) for i in range(T)}}
    with open(os.path.join(args.out, "ntv3_finetune_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nbest test mean Pearson (log1p): {best:.4f} | tracks: {T}")


if __name__ == "__main__":
    main()
