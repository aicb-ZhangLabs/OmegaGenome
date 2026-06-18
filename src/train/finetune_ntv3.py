"""Fine-tune NTv3 on the NTv3 Benchmark bigWig tracks — faithful reproduction of the paper recipe.

Direct port of InstaDeepAI's official notebook
``notebooks_tutorials/03_fine_tuning_posttrained_model_biwig.ipynb`` ("reproduce paper results"),
restructured into our module layout. Every defaulted hyperparameter matches the notebook config, and
the data transform / loss / metric / optimizer schedule are the official ones:

  * data:   dense overlapping windows across ALL split regions (``GenomeBigWigDataset``),
            target = ``x/track_mean`` + smooth-clip (NOT log1p)
  * model:  headless NTv3 ``core`` + fresh ``LinearHead`` over the central 37.5% (``NTv3BigWigModel``)
  * loss:   Poisson-multinomial (shape + scale/5)
  * optim:  AdamW (wd 0.01), warmup -> peak 5e-5 -> square decay
  * metric: torchmetrics PearsonCorrCoef (float64), pooled over positions in the scaled space
  * loop:   step-based with gradient accumulation (eff. batch 32), validate + best-model select

Reference (their PyTorch pipeline, human): mean test Pearson ~0.605.

  python -m src.train.finetune_ntv3 --data_dir <benchmark> --out <dir> [--smoke]
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.ntv3_benchmark import stage_to_local
from src.data.ntv3_ft_data import GenomeBigWigDataset, load_benchmark_frames, make_target_scaling_fn
from src.model.ntv3_finetune import NTV3_CROP_FRAC, NTv3BigWigModel
from src.trainer.ntv3_optim import build_optimizer_and_scheduler
from src.trainer.track_losses import poisson_multinomial_loss
from src.trainer.track_metrics import TracksMetrics


def _parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="local NTv3_benchmark_dataset dir")
    ap.add_argument("--model", default="InstaDeepAI/NTv3_650M_post")
    ap.add_argument("--out", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--stage_dir", default=None, help="node-local dir to stage genome+bigWigs into")
    # data (notebook defaults)
    ap.add_argument("--sequence_length", type=int, default=32768)
    ap.add_argument("--keep_target_center_fraction", type=float, default=NTV3_CROP_FRAC)
    ap.add_argument("--train_overlap", type=float, default=0.999)
    # training (notebook defaults: eff. batch 4*8=32, ~20.9B tokens)
    ap.add_argument("--mini_batch_size", type=int, default=4)
    ap.add_argument("--num_accumulation_gradient", type=int, default=8)
    ap.add_argument("--num_steps_training", type=int, default=19932)
    ap.add_argument("--initial_learning_rate", type=float, default=1e-5)
    ap.add_argument("--end_learning_rate", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_steps_warmup", type=int, default=598)
    ap.add_argument("--log_every_n_steps", type=int, default=50)
    ap.add_argument("--validate_every_n_steps", type=int, default=500)
    ap.add_argument("--num_validation_samples", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_lora", action="store_true", help="cheap LoRA core instead of full fine-tune")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (notebook default)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (few steps / val samples) to validate the pipeline")
    return ap.parse_args()


def _make_loader(fasta, bw_paths, regions, split_regions, args, tokenizer, transform_fn,
                 overlap, limit, shuffle):
    ds = GenomeBigWigDataset(fasta, bw_paths, split_regions, args.sequence_length, tokenizer,
                             transform_fn, overlap=overlap,
                             keep_target_center_fraction=args.keep_target_center_fraction,
                             limit_num_samples=limit)
    return ds, DataLoader(ds, batch_size=args.mini_batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, drop_last=shuffle)


@torch.no_grad()
def _evaluate(model, loader, metrics, device):
    model.eval()
    metrics.reset()
    for batch in loader:
        logits = model(batch["tokens"].to(device))["bigwig_tracks_logits"]
        targets = batch["bigwig_targets"].to(device)
        loss = poisson_multinomial_loss(logits, targets)
        metrics.update(logits, targets, loss.item())
    return metrics.compute()


def main():
    args = _parse_args()
    if args.smoke:  # tiny but exercises every code path
        args.num_steps_training, args.num_steps_warmup = 6, 2
        args.validate_every_n_steps, args.num_validation_samples = 3, 8
        args.num_accumulation_gradient, args.num_workers, args.log_every_n_steps = 2, 2, 2
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    data_dir = args.data_dir
    if args.stage_dir:
        print(f"staging genome+bigWigs node-local -> {args.stage_dir} ...", flush=True)
        data_dir = stage_to_local(args.data_dir, args.stage_dir, args.species)

    fasta, bw_paths, bw_ids, regions_by_split, track_means, track_assays = \
        load_benchmark_frames(data_dir, args.species)
    T = len(bw_ids)
    print(f"tracks={T} seq_len={args.sequence_length} center={args.keep_target_center_fraction} "
          f"device={device} mode={'LoRA' if args.use_lora else 'full-FT'}", flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    transform_fn = make_target_scaling_fn(track_means)

    train_ds, train_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["train"],
                                          args, tokenizer, transform_fn, args.train_overlap, None, True)
    val_ds, val_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["val"],
                                      args, tokenizer, transform_fn, 0.0, args.num_validation_samples, False)
    test_ds, test_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["test"],
                                        args, tokenizer, transform_fn, 0.0, None, False)
    print(f"windows: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}", flush=True)

    model = NTv3BigWigModel(args.model, T, species_str=args.species,
                            keep_target_center_fraction=args.keep_target_center_fraction,
                            use_lora=args.use_lora).to(device)
    if args.compile:
        model = torch.compile(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M", flush=True)

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, args.initial_learning_rate, args.end_learning_rate, args.weight_decay,
        args.num_steps_warmup, args.num_steps_training)
    train_metrics = TracksMetrics(bw_ids, device)
    val_metrics = TracksMetrics(bw_ids, device)

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(project="ntv3_benchmark_finetune_faithful", config=vars(args),
                         name=f"{'lora' if args.use_lora else 'fullft'}-seq{args.sequence_length}")

    best_val = 0.0
    best_path = os.path.join(args.out, "best_model.pth")
    train_iter = iter(train_loader)
    model.train()
    for step in range(args.num_steps_training):
        optimizer.zero_grad()
        for _ in range(args.num_accumulation_gradient):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            logits = model(batch["tokens"].to(device))["bigwig_tracks_logits"]
            targets = batch["bigwig_targets"].to(device)
            loss = poisson_multinomial_loss(logits, targets)
            (loss / args.num_accumulation_gradient).backward()
            train_metrics.update(logits, targets, loss.item())
        optimizer.step()
        scheduler.step()

        if (step + 1) % args.log_every_n_steps == 0:
            m = train_metrics.compute()
            print(f"step {step+1}/{args.num_steps_training}  lr={scheduler.get_last_lr()[0]:.2e}  "
                  f"loss={m['loss']:.4f}  train_pearson={m['mean/pearson']:.4f}", flush=True)
            if run:
                run.log({"step": step + 1, "train_loss": m["loss"], "train_pearson": m["mean/pearson"],
                         "lr": scheduler.get_last_lr()[0]})
            train_metrics.reset()

        if (step + 1) % args.validate_every_n_steps == 0:
            vm = _evaluate(model, val_loader, val_metrics, device)
            print(f"  [val] step {step+1}  mean_pearson={vm['mean/pearson']:.4f}", flush=True)
            if run:
                run.log({"step": step + 1, "val_pearson": vm["mean/pearson"]})
            if vm["mean/pearson"] > best_val:
                best_val = vm["mean/pearson"]
                torch.save(model.state_dict(), best_path)
                print(f"  new best val mean_pearson={best_val:.4f} -> saved", flush=True)
            model.train()

    # final test with the best checkpoint (no test-set selection bias)
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics = TracksMetrics(bw_ids, device)
    tm = _evaluate(model, test_loader, test_metrics, device)

    by_assay = {}
    for tid, assay in zip(bw_ids, track_assays):
        by_assay.setdefault(assay, []).append(tm[f"{tid}/pearson"])
    per_assay = {a: float(np.mean(v)) for a, v in by_assay.items()}
    result = {"model": args.model, "mode": "lora" if args.use_lora else "full-FT",
              "sequence_length": args.sequence_length, "n_tracks": T,
              "best_val_mean_pearson": float(best_val), "test_mean_pearson": float(tm["mean/pearson"]),
              "test_pearson_by_assay": per_assay,
              "per_track_pearson": {tid: tm[f"{tid}/pearson"] for tid in bw_ids}}
    with open(os.path.join(args.out, "ntv3_finetune_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if run:
        run.log({"test_mean_pearson": tm["mean/pearson"], **{f"test/{a}": v for a, v in per_assay.items()}})
        run.finish()
    print(f"\nbest-val {best_val:.4f} -> TEST mean Pearson {tm['mean/pearson']:.4f} | tracks {T}", flush=True)


if __name__ == "__main__":
    main()
