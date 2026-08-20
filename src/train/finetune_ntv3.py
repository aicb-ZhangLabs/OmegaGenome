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
from src.data.ntv3_ft_data import (GenomeBigWigDataset, load_benchmark_frames, make_target_scaling_fn,
                                    validate_subset_bigwigs)
from src.model.ntv3_finetune import NTV3_CROP_FRAC, build_bigwig_model, load_finetuned_bigwig_teacher
from src.trainer.ntv3_optim import build_optimizer_and_scheduler
from src.trainer.track_losses import poisson_multinomial_loss
from src.trainer.track_distill import TrackKDConfig, resolve_kd_track_targets, track_kd_loss
from src.trainer.teacher_cache import cache_teacher_logits
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
    ap.add_argument("--max_test_samples", type=int, default=None,
                    help="cap test windows (default None = all test regions, per notebook; smoke caps it)")
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--max_train_samples", type=int, default=None,
                    help="cap train windows (for a cached-baseline matched to a cache's fixed window set; "
                         "pair with --train_overlap 0). Default None = full stream.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--use_lora", action="store_true", help="cheap LoRA core instead of full fine-tune")
    ap.add_argument("--student_arch", default="ntv3", choices=["ntv3", "bpnet"],
                    help="student architecture: 'ntv3' (pretrained transformer backbone, default) or "
                         "'bpnet' (from-scratch dilated CNN — the genomics-native, tiny/fast student)")
    ap.add_argument("--bpnet_channels", type=int, default=None,
                    help="bpnet student: conv width (default None = stock 64-ch 'original' tower)")
    ap.add_argument("--bpnet_n_dilated", type=int, default=None,
                    help="bpnet student: number of dilated residual blocks (dilation 2^i, uncapped)")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (notebook default)")
    ap.add_argument("--amp", action="store_true",
                    help="bf16 autocast + TF32 matmuls (~1.5-2x faster on Ampere/Hopper, negligible "
                         "Pearson impact). Default off so the faithful fp32 repro is byte-unchanged.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny end-to-end run (few steps / val samples) to validate the pipeline")
    ap.add_argument("--dry_run", action="store_true",
                    help="push ONE batch through train+val+test and exit (no real training) — fastest "
                         "check that every code path runs error-free for a given config")
    # --- Knowledge distillation (set --teacher to switch from plain finetune/baseline to KD) ---
    ap.add_argument("--teacher", default=None,
                    help="finetuned NTv3 bigWig teacher ckpt (e.g. reproduced 650M best_model.pth); "
                         "None = plain finetune (the baseline)")
    ap.add_argument("--teacher_base", default="InstaDeepAI/NTv3_650M_post",
                    help="base arch the teacher ckpt was finetuned from")
    ap.add_argument("--teacher_specialist", action="store_true",
                    help="the teacher was itself trained on the SAME --track_subset (1-track head), so "
                         "load it at the subset size and skip index-selecting (vs a 34-track generalist)")
    ap.add_argument("--cached_teacher_logits", default=None,
                    help="path to a precompute_teacher_logits cache: KD reads teacher logits from it "
                         "(no per-step teacher forward). Fixes the train set to the cache's windows; "
                         "logit-only so feature alignment (w_mse) is forced off.")
    ap.add_argument("--cache_in_ram", action="store_true",
                    help="for a DIR (joint memmap) --cached_teacher_logits: load the whole logits.npy (fp16) "
                         "into RAM once (one sequential read) instead of per-batch mmap reads. Avoids the "
                         "sshfs per-batch random-memmap hang WITHOUT node-local /tmp staging (needs a big-RAM "
                         "node, e.g. laniakea 1TB, for the ~53GB joint-34 cache). Numerically identical.")
    ap.add_argument("--cached_teacher_feat", default=None,
                    help="dir of a cache_teacher_features cache (feat.npy + meta.pt): enables the w_mse "
                         "feature-alignment term from CACHED compressed (PCA d', pooled) teacher features, so "
                         "feature-KD needs no live teacher. Pins the train windows to the cache's set (idx-"
                         "aligned). Use with --cached_teacher_logits (fully cached) or a live --teacher.")
    ap.add_argument("--cache_logits", action="store_true",
                    help="after a TEACHER training (NTv3, no --teacher/--cached), cache the BEST-VAL "
                         "checkpoint's per-track logits to {gen,spec}/t{idx}.pt. Only the best checkpoint "
                         "is cached (intermediate teacher states are not meaningful). Idempotent: skips "
                         "tracks already cached (re-runs that find the cache don't recompute).")
    ap.add_argument("--cache_logits_dir", default=None,
                    help="root for --cache_logits output (default <SSD>/ntv3_cache/teacher_logits)")
    ap.add_argument("--cache_logits_n", type=int, default=64000,
                    help="fixed non-overlapping train windows to cache (caps at the ~63707 available)")
    ap.add_argument("--kd_w_ce", type=float, default=0.5, help="KD ground-truth term weight")
    ap.add_argument("--kd_w_kl", type=float, default=0.5, help="KD distill (teacher) term weight")
    ap.add_argument("--kd_w_mse", type=float, default=0.2, help="KD feature term weight (0 = off here)")
    ap.add_argument("--kd_gt_loss", default="poisson_multinomial",
                    choices=["poisson_multinomial", "mse", "pearson", "dist", "standardized_mse"])
    ap.add_argument("--kd_distill_loss", default="poisson_multinomial",
                    choices=["poisson_multinomial", "mse", "pearson", "teacher_bounded",
                             "dist", "standardized_mse", "cwd"])
    ap.add_argument("--kd_multinomial_weight", type=float, default=5.0)
    ap.add_argument("--kd_cwd_temperature", type=float, default=4.0)
    ap.add_argument("--kd_distill_target_gt_mix", type=float, default=0.0)
    ap.add_argument("--track_subset", default=None,
                    help="Comma-separated track indices to train a SPECIALIST student on a subset (e.g. a "
                         "single track) instead of all 34 — diagnostic for whether capacity-sharing across "
                         "tracks is the bottleneck. The teacher's output is index-selected to match.")
    return ap.parse_args()


def _make_loader(fasta, bw_paths, regions, split_regions, args, tokenizer, transform_fn,
                 overlap, limit, shuffle):
    ds = GenomeBigWigDataset(fasta, bw_paths, split_regions, args.sequence_length, tokenizer,
                             transform_fn, overlap=overlap,
                             keep_target_center_fraction=args.keep_target_center_fraction,
                             limit_num_samples=limit)
    return ds, DataLoader(ds, batch_size=args.mini_batch_size, shuffle=shuffle,
                          num_workers=args.num_workers)  # no drop_last (matches notebook)


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
        args.max_test_samples = 8  # cap test too (full test set is huge; smoke just checks the path)
    if args.dry_run:  # ~one batch through train -> val -> test, then exit: fastest error check for a config
        # 2 train steps / warmup 1 (the scheduler needs num_steps>num_warmup>0: it divides by
        # log(num_steps/num_warmup)); val + test each one batch; no dataloader workers for clear errors.
        args.num_steps_training, args.num_steps_warmup = 2, 1
        args.validate_every_n_steps, args.num_validation_samples = 1, args.mini_batch_size
        args.num_accumulation_gradient, args.num_workers, args.log_every_n_steps = 1, 0, 1
        args.max_test_samples = args.mini_batch_size  # single test batch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.amp:  # TF32 matmul/cudnn (free on Ampere+); bf16 autocast is applied per-step in the loop
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.out, exist_ok=True)

    data_dir = args.data_dir
    if args.stage_dir:
        print(f"staging genome+bigWigs node-local -> {args.stage_dir} ...", flush=True)
        data_dir = stage_to_local(args.data_dir, args.stage_dir, args.species)

    fasta, bw_paths, bw_ids, regions_by_split, track_means, track_assays = \
        load_benchmark_frames(data_dir, args.species)
    # Optional SPECIALIST mode: train on a subset of tracks (e.g. one). Subset everything track-indexed
    # (head size T, bigWig paths/targets, scaling means, metric ids) consistently; keep the ORIGINAL
    # indices to index-select the teacher's full-track output to match in the KD loop.
    native_n = len(bw_ids)  # native track count (e.g. 34) — captured before any subset
    idx = None
    if args.track_subset:
        idx = [int(x) for x in args.track_subset.split(",")]
        bad = [i for i in idx if i < 0 or i >= native_n]
        if bad:  # fail loud: an out-of-range index would silently mis-target or IndexError downstream
            raise ValueError(f"--track_subset {idx} out of range for {native_n} tracks (bad: {bad})")
        # Log idx -> file_id -> assay UNAMBIGUOUSLY (the dataset/--track_subset order, NOT the assay-sorted
        # CSV order) so the run record always shows exactly which track was trained.
        print("TRACK SUBSET (specialist) — dataset index -> file_id -> assay:", flush=True)
        for i in idx:
            print(f"    [{i}] {bw_ids[i]}  ({track_assays[i]})", flush=True)
        bw_ids = [bw_ids[i] for i in idx]
        bw_paths = [bw_paths[i] for i in idx]
        track_means = (track_means[idx] if hasattr(track_means, "__getitem__")
                       and not isinstance(track_means, list) else [track_means[i] for i in idx])
        track_assays = [track_assays[i] for i in idx]
        # RIGOR GUARD #2: the name-order guard in load_benchmark_frames only checks the 34 file *names*
        # exist + match the canonical order. It does NOT verify a track we actually train has REAL data —
        # so a 0-byte placeholder (e.g. the subset-transfer trick) or a truncated/corrupt bigWig would slip
        # through and fail later with a cryptic dataloader `[bwHdrRead]`. Fail fast HERE (on the STAGED path,
        # post-subset), naming the exact track index + file_id, so a missing/placeholder track is impossible
        # to train on silently. Extracted to a shared helper so the cache builders reuse the identical check.
        validate_subset_bigwigs(idx, bw_ids, bw_paths)
        print(f"track-data validation OK: {len(idx)} selected bigWig(s) real + openable", flush=True)
    T = len(bw_ids)
    # Resolve the teacher's track count + index-select for KD (generalist: select the subset out of the
    # 34-track teacher; specialist: teacher already has a T-track head, used directly). Pure helper.
    teacher_num_tracks, _kd_idx = resolve_kd_track_targets(idx, native_n, args.teacher_specialist)
    kd_track_idx = torch.tensor(_kd_idx, device=device) if _kd_idx is not None else None
    if args.teacher_specialist and idx:
        print(f"teacher is a SPECIALIST ({T}-track head) — matched to the student subset", flush=True)
    print(f"tracks={T} seq_len={args.sequence_length} center={args.keep_target_center_fraction} "
          f"device={device} mode={'LoRA' if args.use_lora else 'full-FT'}", flush=True)

    from transformers import AutoTokenizer
    # Random-init scaled tiers (ntv3-4m/8m/30m/...) have no on-disk tokenizer of their own: all NTv3 sizes
    # share the single-nt tokenizer, so resolve those keys to the 8m_pre template snapshot for tokenizer load.
    from src.model.ntv3_finetune import NTV3_SCALED_SIZES, NTV3_TEMPLATE
    from src.model.ntv3_teacher import prepare_local_snapshot
    _tok_src = prepare_local_snapshot(NTV3_TEMPLATE) if args.model in NTV3_SCALED_SIZES else args.model
    tokenizer = AutoTokenizer.from_pretrained(_tok_src, trust_remote_code=True,
                                              local_files_only=args.model in NTV3_SCALED_SIZES)
    transform_fn = make_target_scaling_fn(track_means)

    # CACHED-teacher KD: load precomputed teacher logits; the train set is FIXED to the cache's windows
    # (same construction -> idx aligns), so no per-step teacher forward. Logit-only -> no feature align.
    cached_logits, train_limit = None, args.max_train_samples  # baseline cap; cache overrides below
    kd_w_mse_forced, kd_w_mse_requested = False, args.kd_w_mse  # trace requested-vs-used w_mse (forced=0 case)
    if args.cached_teacher_logits:
        # Two on-disk layouts, one read contract: a DIR is the joint-34 fp16 MEMMAP cache (logits.npy +
        # meta.pt) streamed a batch-slice at a time (too big — ~53GB — for RAM); a FILE is the in-RAM .pt
        # cache (per-track / small). Both yield `cached_logits` supporting `.shape` + `[idx_tensor].to()`,
        # so the KD loop below reads either identically. The live-teacher path (pass --teacher and NOT
        # --cached_teacher_logits) remains the faithful fallback and is untouched.
        if os.path.isdir(args.cached_teacher_logits):
            from src.trainer.teacher_cache import open_memmap_logit_cache
            # in_ram: one sequential read of logits.npy into RAM (no per-batch mmap reads) — the /tmp-free
            # mitigation for the sshfs memmap hang (no node-local staging). Off => streaming mmap reader.
            cached_logits, _cm = open_memmap_logit_cache(args.cached_teacher_logits,
                                                         in_ram=args.cache_in_ram)
            _cache_subset, cache_coords = _cm["track_subset"], _cm["coords"]
            train_limit, args.train_overlap = _cm["limit_num_samples"], _cm.get("overlap", 0.0)
        else:
            if not os.path.exists(args.cached_teacher_logits):
                # cache-along-distillation: build it once from --teacher (cheap single pass), then train cached
                if not args.teacher:
                    raise FileNotFoundError(f"cache {args.cached_teacher_logits} missing and no --teacher to build it")
                print(f"cached_teacher_logits missing -> building from --teacher (cache-along-distillation)", flush=True)
                from src.trainer.teacher_cache import build_cache_from_teacher
                build_cache_from_teacher(args.cached_teacher_logits, args.teacher, args.teacher_base,
                                         args.teacher_specialist, idx, native_n, fasta, bw_paths, bw_ids,
                                         regions_by_split["train"], tokenizer, args.sequence_length,
                                         args.cache_logits_n, device, args.amp)
            _c = torch.load(args.cached_teacher_logits, map_location="cpu")
            cached_logits = _c["logits"]             # [N, L_out, T_subset], float32, idx-aligned
            _cache_subset, cache_coords = _c["track_subset"], _c["coords"]
            # Rebuild the IDENTICAL window set: same limit budget + overlap=0 as the precompute (else the
            # train tiling differs and the cache misaligns — caught by the coord assert below).
            train_limit, args.train_overlap = _c["limit_num_samples"], _c.get("overlap", 0.0)
        if _cache_subset != idx:
            raise ValueError(f"cache track_subset {_cache_subset} != --track_subset {idx}")
        kd_track_idx = None                          # cache is already the subset; no index-select
        if args.kd_w_mse > 0 and not args.cached_teacher_feat:
            print(f"[CONFIG-OVERRIDE] requested --kd_w_mse {args.kd_w_mse} but FORCING w_mse=0.0: logits are "
                  f"cached, features are NOT (no --cached_teacher_feat) -> this is a 2-TERM KD (no feature "
                  f"alignment). Recorded in result.json kd.w_mse_requested / w_mse_forced_to_0.", flush=True)
            kd_w_mse_forced, kd_w_mse_requested = True, args.kd_w_mse
            args.kd_w_mse = 0.0
        print(f"CACHED teacher logits: {tuple(cached_logits.shape)} from {args.cached_teacher_logits}; "
              f"train fixed to {train_limit} windows", flush=True)

    # CACHED-teacher FEATURES (separate `feat/` namespace): the compressed (PCA d', pooled) backbone hidden
    # state, idx-aligned to the same window set, so the w_mse feature-alignment term needs no live teacher.
    cached_feat, feat_coords = None, None
    if args.cached_teacher_feat:
        # GUARD (audit 2026-06-26): cached features only enter the KD loss when KD is active; if neither a
        # live --teacher nor --cached_teacher_logits is given, kd_cfg stays None and the feature cache would
        # be SILENTLY ignored (plain-baseline training). Reject that footgun explicitly instead.
        if not (args.teacher or args.cached_teacher_logits):
            raise ValueError("--cached_teacher_feat needs a teacher source too (--teacher or "
                             "--cached_teacher_logits); alone it would be silently ignored (no KD).")
        _fm = torch.load(os.path.join(args.cached_teacher_feat, "meta.pt"), map_location="cpu")
        cached_feat = np.asarray(np.load(os.path.join(args.cached_teacher_feat, "feat.npy")))  # [N,Lp,d'] fp16 -> RAM
        feat_coords = _fm["coords"]
        if cached_logits is None:  # no logit cache -> pin the train window set from the FEATURE cache instead
            # use the ORIGINAL limit (n_windows), NOT num_windows: sample_regions_for_total_length keys off
            # the limit, so limit=num_windows would resample DIFFERENT regions (off-by-N). Mirror the logit cache.
            train_limit = _fm.get("limit_num_samples", _fm["num_windows"])
            args.train_overlap = _fm.get("overlap", 0.0)
        elif cached_feat.shape[0] != cached_logits.shape[0]:
            raise RuntimeError(f"feat cache N {cached_feat.shape[0]} != logit cache N {cached_logits.shape[0]}")
        print(f"CACHED teacher features: {cached_feat.shape} d'={_fm['d_prime']} pool={_fm['pool']} "
              f"evr={_fm['explained_var_ratio']:.4f} from {args.cached_teacher_feat}", flush=True)

    train_ds, train_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["train"],
                                          args, tokenizer, transform_fn, args.train_overlap, train_limit, True)
    val_ds, val_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["val"],
                                      args, tokenizer, transform_fn, 0.0, args.num_validation_samples, False)
    test_ds, test_loader = _make_loader(fasta, bw_paths, regions_by_split, regions_by_split["test"],
                                        args, tokenizer, transform_fn, 0.0, args.max_test_samples, False)
    print(f"windows: train {len(train_ds)} / val {len(val_ds)} / test {len(test_ds)}", flush=True)
    if cached_logits is not None:  # RIGOR: the train windows MUST match the cache, else KD is misaligned
        if len(train_ds) != cached_logits.shape[0]:
            raise RuntimeError(f"cached KD: train windows {len(train_ds)} != cache {cached_logits.shape[0]}")
        for j in (0, len(train_ds) // 2, len(train_ds) - 1):  # spot-check coord alignment
            ci = max(0, np.searchsorted(train_ds._cumulative_starts, j, side="right") - 1)
            ri = train_ds.region_info[ci]
            start = ri["region_start_offset"] + (j - train_ds._cumulative_starts[ci]) * train_ds.stride
            if (ri["chr_name"], int(start)) != tuple(cache_coords[j]):
                raise RuntimeError(f"cached KD: window {j} {(ri['chr_name'], int(start))} != cache {cache_coords[j]}")
        print(f"cached-teacher alignment verified ({len(train_ds)} windows)", flush=True)
    if cached_feat is not None:  # same RIGOR for the feature cache (idx must map to the same windows)
        if len(train_ds) != cached_feat.shape[0]:
            raise RuntimeError(f"cached feat: train windows {len(train_ds)} != cache {cached_feat.shape[0]}")
        for j in (0, len(train_ds) // 2, len(train_ds) - 1):
            ci = max(0, np.searchsorted(train_ds._cumulative_starts, j, side="right") - 1)
            ri = train_ds.region_info[ci]
            start = ri["region_start_offset"] + (j - train_ds._cumulative_starts[ci]) * train_ds.stride
            if (ri["chr_name"], int(start)) != tuple(feat_coords[j]):
                raise RuntimeError(f"cached feat: window {j} {(ri['chr_name'], int(start))} != {feat_coords[j]}")
        print(f"cached-feature alignment verified ({len(train_ds)} windows)", flush=True)

    model = build_bigwig_model(args.model, T, species_str=args.species,
                               keep_target_center_fraction=args.keep_target_center_fraction,
                               use_lora=args.use_lora, student_arch=args.student_arch,
                               channels=args.bpnet_channels, n_dilated=args.bpnet_n_dilated,
                               nuc_ids={t: tokenizer.convert_tokens_to_ids(t) for t in "ACGT"},
                               vocab_size=tokenizer.vocab_size).to(device)
    if args.compile:
        model = torch.compile(model)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.1f}M", flush=True)

    # KD mode: load the frozen finetuned teacher (live KD) OR use cached logits (no teacher model). Either
    # way build the 3-term loss config.
    teacher, kd_cfg = None, None
    if args.teacher or cached_logits is not None:
        if args.teacher and cached_logits is None:  # live teacher each step; cache mode needs NO live model
            teacher = load_finetuned_bigwig_teacher(args.teacher, args.teacher_base, teacher_num_tracks, device=device)
        kd_cfg = TrackKDConfig(w_ce=args.kd_w_ce, w_kl=args.kd_w_kl, w_mse=args.kd_w_mse,
                               gt_loss=args.kd_gt_loss, distill_loss=args.kd_distill_loss,
                               multinomial_weight=args.kd_multinomial_weight,
                               cwd_temperature=args.kd_cwd_temperature,
                               distill_target_gt_mix=args.kd_distill_target_gt_mix)
        print(f"KD: teacher={args.teacher} | w_ce/kl/mse={kd_cfg.w_ce}/{kd_cfg.w_kl}/{kd_cfg.w_mse} "
              f"gt={kd_cfg.gt_loss} distill={kd_cfg.distill_loss}", flush=True)

    # FitNets-style feature alignment for the w_mse term: a student->teacher linear projector built
    # lazily on the first KD batch (once feature dims are known), with its OWN small AdamW (an auxiliary
    # adapter at constant lr, decoupled from the main scheduler). Active only when KD is on AND w_mse>0.
    # Training-only: NOT part of the student, so eval / best_model.pth are unaffected. Live extraction
    # (teacher already runs each step) — no feature cache (65M windows make caching infeasible).
    feat_proj, feat_opt = None, None

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
    start_step = 0
    best_path = os.path.join(args.out, "best_model.pth")
    latest_path = os.path.join(args.out, "latest_state.pth")
    # Resume (for --requeue / crash recovery): restore model+optimizer+scheduler+step+best_val.
    # Pure bookkeeping — the per-step training math is unchanged; training continues its trajectory.
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device)
        # `torch.compile` wraps the model so its state_dict keys gain an `_orig_mod.` prefix. A checkpoint
        # saved on a compiled run (voyager/laniakea) but RESUMED on an eager build (galaxy, NO_COMPILE=1 —
        # galaxy GLIBC can't run triton) would therefore have every real key prefix-mismatched, making
        # load_state_dict(strict=False) report ALL params as "missing" and trip the guard below. Normalize
        # the prefix in BOTH directions so resume is compile-agnostic: strip `_orig_mod.` if the live model
        # is eager, or add it if the live model is compiled but the ckpt was eager.
        _sd = ckpt["model"]
        _model_compiled = any(k.startswith("_orig_mod.") for k in model.state_dict())
        _ckpt_compiled = any(k.startswith("_orig_mod.") for k in _sd)
        if _ckpt_compiled and not _model_compiled:
            _sd = {k[len("_orig_mod."):]: v for k, v in _sd.items()}
        elif _model_compiled and not _ckpt_compiled:
            _sd = {f"_orig_mod.{k}": v for k, v in _sd.items()}
        # strict=False: the saved state_dict carries rotary_embedding cos_cached/sin_cached buffers that
        # a freshly-built (pre-forward) model doesn't register, so they appear as "unexpected keys" and
        # crash a strict load. They are derived buffers (recomputed from positions on first forward), so
        # skipping them is safe; all trainable params + persistent buffers still load. Guard that nothing
        # REAL is missing (only rotary-cache keys may be).
        _info = model.load_state_dict(_sd, strict=False)
        _real_missing = [k for k in _info.missing_keys
                         if not any(t in k for t in ("rotary", "cos_cached", "sin_cached"))]
        if _real_missing:
            raise RuntimeError(f"resume: real missing keys (not rotary cache): {_real_missing[:8]}")
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "feat_proj" in ckpt:  # rebuild the FitNets projector from saved shape, then restore it
            w = ckpt["feat_proj"]["weight"]  # [teacher_dim, student_dim]
            feat_proj = torch.nn.Linear(w.shape[1], w.shape[0]).to(device)
            feat_proj.load_state_dict(ckpt["feat_proj"])
            feat_opt = torch.optim.AdamW(feat_proj.parameters(), lr=args.initial_learning_rate)
            feat_opt.load_state_dict(ckpt["feat_opt"])
        start_step, best_val = ckpt["step"], ckpt["best_val"]
        print(f"RESUMED from {latest_path} at step {start_step} (best_val={best_val:.4f}); "
              f"load skipped {len(_info.unexpected_keys)} rotary-cache keys", flush=True)

    def _robust_torch_save(obj, path, retries=4):
        """torch.save hardened against transient sshfs write blips (the cause of a lost 6h run):
        write to a .tmp sibling, retry on failure, then atomic os.replace. Never leaves a corrupt target."""
        import time as _t
        tmp = path + ".tmp"
        last = None
        for i in range(retries):
            try:
                torch.save(obj, tmp)
                os.replace(tmp, path)        # atomic on same fs; readers never see a partial file
                return
            except (RuntimeError, OSError) as e:
                last = e
                try: os.path.exists(tmp) and os.remove(tmp)
                except OSError: pass
                print(f"[save-retry {i+1}/{retries}] {path}: {e}", flush=True)
                _t.sleep(5 * (i + 1))         # back off; sshfs blips are usually brief
        raise RuntimeError(f"checkpoint save failed after {retries} retries: {path}") from last

    def _save_latest(step):
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "step": step, "best_val": best_val}
        if feat_proj is not None:  # persist the FitNets projector + its optimizer for exact resume
            state["feat_proj"] = feat_proj.state_dict()
            state["feat_opt"] = feat_opt.state_dict()
        _robust_torch_save(state, latest_path)

    train_iter = iter(train_loader)
    model.train()
    for step in range(start_step, args.num_steps_training):
        optimizer.zero_grad()
        if feat_opt is not None:
            feat_opt.zero_grad()
        for _ in range(args.num_accumulation_gradient):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            tokens = batch["tokens"].to(device)
            targets = batch["bigwig_targets"].to(device)
            # bf16 autocast over the forward + loss only (backward inherits dtypes); no GradScaler needed
            # for bf16. enabled=args.amp -> a no-op fp32 path when off, so the faithful repro is unchanged.
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                if kd_cfg is not None:  # KD: gt + teacher-logit distill + (optional) FitNets feature align
                    s_out = model(tokens)
                    logits, s_feat = s_out["bigwig_tracks_logits"], s_out["features"]
                    if cached_logits is not None:  # read precomputed teacher logits for THESE windows (no fwd)
                        teacher_logits, t_feat = cached_logits[batch["idx"]].to(device), None
                    else:
                        with torch.no_grad():
                            t_out = teacher(tokens)
                            teacher_logits, t_feat = t_out["bigwig_tracks_logits"], t_out["features"]
                            if kd_track_idx is not None:  # specialist: match teacher to the student's subset
                                teacher_logits = teacher_logits.index_select(-1, kd_track_idx)
                    if cached_feat is not None:  # cached COMPRESSED teacher feature -> the feature-align target
                        t_feat = torch.from_numpy(cached_feat[batch["idx"].numpy()]).to(device).float()
                    s_feat_proj = None
                    if kd_cfg.w_mse > 0:  # feature alignment: project student emb -> teacher emb, then MSE
                        if feat_proj is None:  # lazy build once feature dims are known
                            feat_proj = torch.nn.Linear(s_feat.shape[-1], t_feat.shape[-1]).to(device)
                            feat_opt = torch.optim.AdamW(feat_proj.parameters(), lr=args.initial_learning_rate)
                            print(f"KD feature alignment ON (FitNets): student {s_feat.shape[-1]} -> "
                                  f"teacher {t_feat.shape[-1]}", flush=True)
                        s_feat_proj = feat_proj(s_feat)
                    loss, _ = track_kd_loss(
                        logits, teacher_logits, targets, cfg=kd_cfg, student_layout="BLT",
                        student_feat=s_feat_proj, teacher_feat=(t_feat if kd_cfg.w_mse > 0 else None))
                else:
                    logits = model(tokens)["bigwig_tracks_logits"]
                    loss = poisson_multinomial_loss(logits, targets)
            (loss / args.num_accumulation_gradient).backward()
            train_metrics.update(logits, targets, loss.item())
        optimizer.step()
        if feat_opt is not None:
            feat_opt.step()
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
                _robust_torch_save(model.state_dict(), best_path)
                print(f"  new best val mean_pearson={best_val:.4f} -> saved", flush=True)
            _save_latest(step + 1)  # periodic full-state checkpoint for resume (every validation)
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
    # Full reproduction record: every CLI arg, the resolved KD term weights/losses, and the code commit.
    # So any run's result.json alone is enough to re-launch the identical experiment (no log archaeology).
    def _git_sha():
        try:
            import subprocess
            return subprocess.check_output(["git", "-C", os.path.dirname(os.path.abspath(__file__)),
                                            "rev-parse", "--short", "HEAD"], text=True).strip()
        except Exception:
            return None
    kd_record = None if kd_cfg is None else {
        "w_ce": kd_cfg.w_ce, "w_kl": kd_cfg.w_kl, "w_mse": kd_cfg.w_mse,
        "gt_loss": kd_cfg.gt_loss, "distill_loss": kd_cfg.distill_loss,
        "feature_alignment_active": bool(kd_cfg.w_mse > 0),  # the 2-term vs 3-term flag, explicitly
        "w_mse_requested": kd_w_mse_requested,               # what the CLI asked for (may differ from w_mse)
        "w_mse_forced_to_0": kd_w_mse_forced,                # True = requested>0 but no cached feat -> forced 2-term
        "multinomial_weight": kd_cfg.multinomial_weight, "cwd_temperature": kd_cfg.cwd_temperature,
        "distill_target_gt_mix": kd_cfg.distill_target_gt_mix, "teacher": args.teacher,
        "teacher_specialist": bool(args.teacher_specialist),
        "cached_teacher_logits": getattr(args, "cached_teacher_logits", None)}
    result = {"model": args.model, "mode": "lora" if args.use_lora else "full-FT",
              "sequence_length": args.sequence_length, "n_tracks": T,
              # DATA-STREAM identity (added 2026-07-24): the realised train window count together with the
              # RESOLVED overlap/limit (config below records them POST-override, i.e. what actually ran).
              # Two arms are data-matched iff these three agree — provable from result.json alone, without
              # re-deriving the tiling. (A cached-teacher run forces overlap=0 + limit=cache's
              # limit_num_samples; a teacher-free run keeps whatever --train_overlap/--max_train_samples say.)
              "n_train_windows": int(len(train_ds)),
              "n_val_windows": int(len(val_ds)), "n_test_windows": int(len(test_ds)),
              # self-documenting track identity (so a result is never ambiguous about WHICH track):
              "track_subset_idx": idx, "track_ids": list(bw_ids), "track_assays": list(track_assays),
              "best_val_mean_pearson": float(best_val), "test_mean_pearson": float(tm["mean/pearson"]),
              "test_pearson_by_assay": per_assay,
              "per_track_pearson": {tid: tm[f"{tid}/pearson"] for tid in bw_ids},
              # === reproduction config (added 2026-06-26) ===
              "kd": kd_record, "git_commit": _git_sha(), "config": vars(args)}
    with open(os.path.join(args.out, "ntv3_finetune_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if run:
        run.log({"test_mean_pearson": tm["mean/pearson"], **{f"test/{a}": v for a, v in per_assay.items()}})
        run.finish()
    print(f"\nbest-val {best_val:.4f} -> TEST mean Pearson {tm['mean/pearson']:.4f} | tracks {T}", flush=True)

    # Cache the BEST-VAL teacher checkpoint's per-track logits (model was just reloaded from best_path
    # above), so later KD students read them instead of forwarding the teacher. ONLY the best checkpoint is
    # cached — intermediate states during training are not a meaningful teacher. Idempotent.
    if args.cache_logits and args.student_arch == "ntv3" and teacher is None and cached_logits is None:
        if not os.path.exists(best_path):
            print("cache_logits: no best checkpoint (no val improvement) -> skipping cache", flush=True)
        else:
            cdir = args.cache_logits_dir or os.path.join(os.path.dirname(args.out.rstrip("/")), "..",
                                                          "ntv3_cache", "teacher_logits")
            cache_teacher_logits(model, tokenizer, fasta, bw_paths, regions_by_split["train"], bw_ids, idx,
                                 cdir, args.cache_logits_n, args.sequence_length, device, args.amp,
                                 teacher_tag=f"best-ckpt:{args.out}")


if __name__ == "__main__":
    main()
