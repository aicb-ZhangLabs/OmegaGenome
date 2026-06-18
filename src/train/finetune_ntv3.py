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

from src.data.ntv3_benchmark import BenchmarkData, load_splits, load_track_meta, sample_windows, stage_to_local
from src.model.ntv3_finetune import NTV3_CROP_FRAC, NTv3FineTune, apply_lora
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
    sp_full = teacher.model.encode_species(["human"] * bs).to(device)  # constant -> compute once
    preds, tgts = [], []
    for i in range(0, len(ids), bs):
        b = ids[i : i + bs].to(device)
        out = model(b, sp_full[: b.shape[0]])  # [B, T, L_out]
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
    ap.add_argument("--lora", action="store_true", help="LoRA fine-tune (cheap; ~1%% params) vs full-FT")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--stage_dir", default=None,
                    help="node-local dir to stage genome+bigWigs into (fast prep; e.g. $SLURM_TMPDIR)")
    ap.add_argument("--prep_only", action="store_true",
                    help="build + cache the prepared data, then exit (run on a fast-IO node like galaxy)")
    ap.add_argument("--wandb", action="store_true", help="log train/val/test curves to wandb (online)")
    args = ap.parse_args()
    if args.lora and args.freeze_backbone:
        raise SystemExit("--lora and --freeze_backbone are mutually exclusive (LoRA needs trainable adapters)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = load_splits(os.path.join(args.data_dir, "human/splits.bed"))
    meta = load_track_meta(os.path.join(args.data_dir, "benchmark_metadata.tsv"), "human")
    T = len(meta)
    crop_len = round(args.window * NTV3_CROP_FRAC)
    print(f"tracks={T} window={args.window} crop_len(L_out)={crop_len} device={device} "
          f"mode={'head-only' if args.freeze_backbone else 'full-FT'}")

    teacher = NTv3Teacher(NTv3TeacherConfig(model_name_or_path=args.model, species="human"), device=device)

    # Cache the prepared (tokenized seqs + bigWig targets) — slow over sshfs, and identical across
    # head-only/LoRA/full-FT/model-size (same NTv3 tokenizer + same windows/tracks). Shared by config.
    cache = os.path.join(args.data_dir, f"_prep_tvt_w{args.window}_tr{args.n_train}_te{args.n_test}.pt")
    if os.path.exists(cache):
        print(f"loading prepared-data cache: {cache}")
        d = torch.load(cache, weights_only=False)
        tr_ids, tr_tgt, va_ids, va_tgt, te_ids, te_tgt = (d["tr_ids"], d["tr_tgt"], d["va_ids"],
                                                          d["va_tgt"], d["te_ids"], d["te_tgt"])
    else:
        read_dir = args.data_dir
        if args.stage_dir:
            print(f"staging genome+bigWigs node-local -> {args.stage_dir} (fast prep) ...")
            read_dir = stage_to_local(args.data_dir, args.stage_dir, "human")
        bd = BenchmarkData(os.path.join(read_dir, "human/genome.fasta"),
                           os.path.join(read_dir, "human/functional_tracks"), meta)
        prep = lambda split, n: _prepare(teacher, bd, sample_windows(splits[split], args.window, n=n),
                                         args.window, crop_len, log1p_targets=False)
        tr_ids, tr_tgt = prep("train", args.n_train)
        va_ids, va_tgt = prep("val", args.n_test)   # val for model selection (their split)
        te_ids, te_tgt = prep("test", args.n_test)  # test for final report only
        torch.save({"tr_ids": tr_ids, "tr_tgt": tr_tgt, "va_ids": va_ids, "va_tgt": va_tgt,
                    "te_ids": te_ids, "te_tgt": te_tgt}, cache)
        print(f"cached prepared data -> {cache}")
    print(f"train {len(tr_ids)} / val {len(va_ids)} / test {len(te_ids)} windows")

    if args.prep_only:
        print("--prep_only: prepared-data cache written; exiting before model build / training.")
        return

    # build the model only now (prep above needs just the tokenizer; this lets prep run on a
    # fast-IO node like galaxy without touching the GPU)
    backbone = teacher.model
    if args.lora:
        backbone = apply_lora(backbone, r=args.lora_r, alpha=args.lora_alpha)
    model = NTv3FineTune(backbone, teacher.model.config.embed_dim, T, freeze_backbone=args.freeze_backbone).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train_params/1e6:.2f}M  ({'LoRA' if args.lora else 'head-only' if args.freeze_backbone else 'full-FT'})")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    os.makedirs(args.out, exist_ok=True)
    run = None
    if args.wandb:
        import wandb

        run = wandb.init(project="ntv3_benchmark_finetune", config=vars(args),
                         name=f"{'lora-r%d' % args.lora_r if args.lora else 'head' if args.freeze_backbone else 'fullft'}")
    sp_full = teacher.model.encode_species(["human"] * args.batch_size).to(device)  # hoisted (constant)
    best_val = -1.0
    best_state = None  # trainable params (LoRA adapters + head) at the best-VAL epoch
    perm = torch.randperm(len(tr_ids))
    for ep in range(args.epochs):
        model.train()
        tot = 0.0
        for i in range(0, len(tr_ids), args.batch_size):
            idx = perm[i : i + args.batch_size]
            b = tr_ids[idx].to(device)
            opt.zero_grad()
            loss = track_distill_loss(model(b, sp_full[: b.shape[0]]), tr_tgt[idx].to(device), args.loss)
            loss.backward()
            opt.step()
            tot += loss.item()
        train_loss = tot / (len(tr_ids) / args.batch_size)
        val_r, _ = _evaluate(model, teacher, va_ids, va_tgt, device, args.batch_size)  # select on VAL
        print(f"epoch {ep:3d}  train_loss={train_loss:.4f}  val_mean_pearson={val_r:.4f}", flush=True)
        if run:
            run.log({"epoch": ep, "train_loss": train_loss, "val_pearson": val_r})
        if val_r > best_val:
            best_val = val_r
            best_state = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    # restore best-VAL checkpoint, then report on TEST (no test-set selection bias)
    if best_state is not None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in best_state:
                    p.copy_(best_state[n].to(device))
        torch.save(best_state, os.path.join(args.out, "trainable_best.pt"))
    test_r, per = _evaluate(model, teacher, te_ids, te_tgt, device, args.batch_size)
    # aggregate by assay for direct comparison to the paper CSV (ATAC/Histone/RNA/eCLIP/PRO-cap)
    by_assay = {}
    for i in range(T):
        if not np.isnan(per[i]):
            by_assay.setdefault(meta[i].assay, []).append(per[i])
    per_assay = {a: float(np.mean(v)) for a, v in by_assay.items()}
    result = {"model": args.model, "mode": "lora" if args.lora else "head" if args.freeze_backbone else "full-FT",
              "lora_r": args.lora_r if args.lora else None, "loss": args.loss, "n_tracks": T,
              "best_val_mean_pearson": best_val, "test_mean_pearson": float(test_r),
              "test_pearson_by_assay": per_assay,
              "per_track_pearson": {meta[i].file_id: (None if np.isnan(per[i]) else float(per[i])) for i in range(T)}}
    with open(os.path.join(args.out, "ntv3_finetune_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if run:
        run.log({"test_mean_pearson": float(test_r), **{f"test/{a}": v for a, v in per_assay.items()}})
        run.finish()
    print(f"\nbest-val mean Pearson {best_val:.4f} -> TEST mean Pearson (log1p) {test_r:.4f} | tracks: {T}")


if __name__ == "__main__":
    main()
