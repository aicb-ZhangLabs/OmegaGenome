import os
import csv
import time
import json
import torch
import numpy as np
import wandb
import shutil
import uuid

from datetime import datetime
from dataclasses import asdict, dataclass
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import PreTrainedTokenizer
from typing import List, Optional
from ..model.distillation import DistillationModel, DistillationModelConfig
from .utils import precompute_teacher_logits
from ..data.dataset import (
    SeqDataset,
)


def create_run_hyperparams_str(config: DistillationModelConfig) -> str:
    config_dict = asdict(config)
    sorted_keys = sorted(config_dict.keys())
    return "_".join([f"{k}{config_dict[k]}" for k in sorted_keys])


def evaluate(model, loader, device):
    """Evaluate model and return metrics as Python native types"""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch[0].to(device)
            labs = batch[1].to(device)
            logits = model(ids)
            p = logits.argmax(dim=-1).cpu().numpy()
            preds.extend(p)
            labels.extend(labs.cpu().numpy())
    preds = np.array(preds)
    labels = np.array(labels)

    # Convert numpy types to Python native types for JSON serialization
    return {
        "f1": float(f1_score(labels, preds, average="macro")),
        "mcc": float(matthews_corrcoef(labels, preds)),
    }


def create_run_directory(parent_dir, task_name, config: DistillationModelConfig, dry_run=False):
    """
    Creates a systematic directory structure:
    {parent_dir}/{task_name}/{uuid}/{date}_CE{ce}_KL{kl}_MSE{mse}_T{temp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparam_str = create_run_hyperparams_str(config)
    prefix = "DRYRUN_" if dry_run else ""
    run_dir = os.path.join(
        parent_dir, task_name, f"{prefix}{timestamp}/{uuid.uuid4()}/{hyperparam_str}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


# Filesystem-error fragments that indicate a *transient* failure (e.g. an sshfs blip on the
# shared galaxy SSD that makes torch.save raise "File ... cannot be opened"). These resolve in
# seconds, so retrying recovers the run. Anything NOT matching (CUDA OOM, shape mismatch, ...)
# is a real bug and must surface immediately, so we never retry it.
_TRANSIENT_FS_SIGNATURES = (
    "cannot be opened",
    "unable to open file",
    "Input/output error",
    "Transport endpoint is not connected",
    "No such file or directory",
    "Stale file handle",
    "Resource temporarily unavailable",
)


def _is_transient_fs_error(err: BaseException) -> bool:
    """True if ``err`` looks like a recoverable filesystem hiccup (sshfs blip), not a real bug.

    Any ``OSError`` qualifies; a ``RuntimeError`` qualifies only if its message matches a known
    transient-FS signature (so a CUDA/shape ``RuntimeError`` is NOT swallowed and re-raises fast).
    """
    if isinstance(err, OSError):
        return True
    msg = str(err)
    return any(sig in msg for sig in _TRANSIENT_FS_SIGNATURES)


def _retry_io(fn, what: str, attempts: int = 4, base_delay: float = 3.0):
    """Run idempotent I/O closure ``fn``, retrying ONLY transient FS errors with exponential backoff.

    Recovers transient sshfs disconnects on the shared galaxy SSD (the cause of the carbon-distill
    checkpoint-save failures) — a blip clears in seconds, so backoff 3/9/27s converts the failure
    into a success without losing the run. Non-transient errors re-raise immediately (no masking).
    Re-raises the last error if every attempt fails. ``what`` labels the op in the retry log.
    """
    last = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except (RuntimeError, OSError) as e:
            if not _is_transient_fs_error(e):
                raise  # real bug -> surface now, don't waste retries
            last = e
            if attempt == attempts:
                break
            delay = base_delay * (3 ** (attempt - 1))
            print(
                f"[io-retry] {what}: attempt {attempt}/{attempts} hit transient FS error "
                f"({type(e).__name__}: {e}); retrying in {delay:.0f}s",
                flush=True,
            )
            time.sleep(delay)
    raise last


def save_checkpoint(model, epoch, val_mcc, run_dir, is_best=False):
    """
    Save checkpoint with systematic naming: epoch_{num}_valmcc_{score}
    Also maintains a 'best_model' directory for the best checkpoint

    The whole write sequence is wrapped in ``_retry_io`` so a transient sshfs blip on the shared
    galaxy SSD (which otherwise raises "student.pt cannot be opened" and FAILs the job) is retried
    instead of crashing. The closure is idempotent — re-running re-creates the dir and overwrites.
    """
    # Format MCC to 4 decimal places
    mcc_str = f"{val_mcc:.4f}".replace(".", "p")  # Replace . with p for filename
    epoch_dir = os.path.join(run_dir, f"epoch_{epoch}_valmcc_{mcc_str}")

    def _do_save():
        os.makedirs(epoch_dir, exist_ok=True)

        # Save model
        model_path = os.path.join(epoch_dir, "student.pt")
        torch.save(model.state_dict(), model_path)

        # Save metadata
        metadata = {
            "epoch": int(epoch),  # Ensure Python int
            "val_mcc": float(val_mcc),  # Ensure Python float
            "timestamp": datetime.now().isoformat(),
        }
        with open(os.path.join(epoch_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        # If this is the best model, copy to best_model directory
        if is_best:
            best_dir = os.path.join(run_dir, "best_model")
            if os.path.exists(best_dir):
                shutil.rmtree(best_dir)
            shutil.copytree(epoch_dir, best_dir)

            # Also save a reference file
            with open(os.path.join(run_dir, "best_model_info.txt"), "w") as f:
                f.write(f"Best model: epoch {epoch}, val_mcc {val_mcc:.4f}\n")
                f.write(f"Location: {epoch_dir}\n")

    _retry_io(_do_save, what=f"save_checkpoint epoch {epoch} -> {run_dir}")
    return epoch_dir


def _early_stop_step(is_best: bool, epochs_no_improve: int, patience):
    """One epoch of early-stop bookkeeping (pure -> unit-testable).

    - ``is_best``: did this epoch set a new best val metric?
    - ``epochs_no_improve``: consecutive non-improving epochs BEFORE this one.
    - ``patience``: stop after this many consecutive non-improving epochs. None or <=0 -> disabled.

    Returns ``(epochs_no_improve, should_stop)``. A new best resets the counter to 0 (so a late
    improvement after a long plateau keeps training); ``should_stop`` is True only once the counter
    REACHES ``patience``. Because the trainer reports the best-val checkpoint, stopping here yields the
    identical reported metric with less compute.
    """
    epochs_no_improve = 0 if is_best else epochs_no_improve + 1
    should_stop = bool(patience and patience > 0 and epochs_no_improve >= patience)
    return epochs_no_improve, should_stop


@dataclass
class DistillTrainerConfig:
    output_dir: str
    wandb_project: str
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    max_len: int = 1024
    log_batch_every: int = 50
    eval_every_n_epochs: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Base dir for the precompute logits/features cache. None -> project_path (legacy /extra). Set to
    # a fast local disk (SSD) to keep the cache off the degraded /extra NFS.
    cache_base_dir: Optional[str] = None
    # Batch size for the TEACHER forward (precompute + teacher eval). None -> batch_size. A large
    # teacher (Carbon-3B) at seq~1000 needs a small batch (attention is O(seq^2)); the student trains
    # at the larger batch_size separately.
    teacher_batch_size: Optional[int] = None
    # Early stopping: stop if val_mcc hasn't improved for this many epochs. None -> disabled (run all
    # epochs). Since we report the BEST-VAL checkpoint, early-stopping yields the SAME reported MCC with
    # less compute — used for the HP search; the final best-HP runs keep it None (full schedule).
    early_stop_patience: Optional[int] = None


def train_distill_task(
    config: DistillTrainerConfig,
    distillation_config: DistillationModelConfig,
    task_name: str,
    teacher_tokenizer: PreTrainedTokenizer,
    teacher_model: torch.nn.Module,
    model: torch.nn.Module,
    distillation_model: DistillationModel,
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    X_test: List[str],
    y_test: List[int],
    run_dir: str,
    teacher_parent_dir: str,
    teacher_ckpt: str,  # <-- ADD THIS PARAMETER
    resume_from_checkpoint: Optional[str] = None,  # NEW parameter
    resume_from_epoch: int = 0,  # NEW parameter
    input_prefix: str = "",  # teacher input formatting (Carbon "<dna>"); no-op default for others
    add_special_tokens: bool = True,
):
    # move models to device
    model.to(config.device)
    teacher_model.to(config.device)
    distillation_model.to(config.device)

    # ===== PRECOMPUTE TEACHER OUTPUTS (with caching) =====
    needs_logits = distillation_config.weight_kl > 0
    needs_features = distillation_config.weight_mse > 0

    train_tlogits = None
    train_tfeatures = None

    # Only precompute if we need logits or features
    if needs_logits or needs_features:
        if needs_logits and needs_features:
            print("Precomputing teacher logits and features...")
        elif needs_features:
            print("Precomputing teacher features (weight_kl=0)...")
        else:
            print("Precomputing teacher logits...")

        from config.env import project_path

        cache_base = config.cache_base_dir or project_path  # SSD when set, else legacy /extra
        teacher_bs = config.teacher_batch_size or config.batch_size  # small batch for the 3B forward
        train_tlogits, train_tfeatures = precompute_teacher_logits(
            teacher_tokenizer,
            teacher_model,
            X_train,
            teacher_bs,
            config.device,
            config.max_len,
            needs_logits=needs_logits,  # NEW: explicit logits flag
            needs_features=needs_features,
            input_prefix=input_prefix,
            add_special_tokens=add_special_tokens,
            # Cache parameters
            project_path=cache_base,
            teacher_parent_dir=teacher_parent_dir,
            task_name=task_name,
            teacher_ckpt=teacher_ckpt,
            use_cache=True,
        )
        print("Teacher outputs precomputed.")
    else:
        print("Skipping teacher precomputation (weight_kl=0 and weight_mse=0)")

    train_ds = SeqDataset(X_train, y_train, config.max_len, train_tlogits, train_tfeatures)
    val_ds = SeqDataset(X_val, y_val, config.max_len)
    test_ds = SeqDataset(X_test, y_test, config.max_len)

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4)

    best_val_mcc = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    global_step = 0
    start_time = time.time()

    # Training history
    training_history = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    start_epoch = 1
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint_path = os.path.join(resume_from_checkpoint, "student.pt")
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.device))
        start_epoch = resume_from_epoch + 1
        print(f"Resuming from epoch {start_epoch}")
    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            loss, metrics = distillation_model(batch)
            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss.item()
            if global_step % config.log_batch_every == 0:
                log_dict = {
                    "train/batch_loss": loss.item(),
                }
                log_dict.update({f"train/{k}": v for k, v in metrics.items()})
                wandb.log(log_dict, step=global_step)

        # Epoch-end evaluation
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, config.device)

        # Determine if this is the best model
        is_best = val_metrics["mcc"] > best_val_mcc
        epochs_no_improve, should_stop = _early_stop_step(
            is_best, epochs_no_improve, config.early_stop_patience
        )
        if is_best:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch

        # Save checkpoint for this epoch
        checkpoint_dir = save_checkpoint(model, epoch, val_metrics["mcc"], run_dir, is_best=is_best)

        # Record epoch info - ensure all values are JSON serializable
        epoch_info = {
            "epoch": int(epoch),
            "train_loss": float(avg_loss),
            "val_f1": float(val_metrics["f1"]),
            "val_mcc": float(val_metrics["mcc"]),
            "is_best": bool(is_best),  # Convert numpy bool to Python bool
            "checkpoint_dir": str(checkpoint_dir),
        }

        # Test evaluation every N epochs
        if epoch % config.eval_every_n_epochs == 0:
            test_ep = evaluate(model, test_loader, config.device)
            epoch_info["test_mcc"] = float(test_ep["mcc"])
            epoch_info["test_f1"] = float(test_ep["f1"])
            wandb.log({f"test_epoch_{epoch}_mcc": test_ep["mcc"]}, step=global_step)
            print(f"[{task_name}] Test @epoch {epoch} mcc {test_ep['mcc']:.4f}")

        training_history.append(epoch_info)

        # Logging
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": avg_loss,
                "val/f1": val_metrics["f1"],
                "val/mcc": val_metrics["mcc"],
                "time/elapsed_s": time.time() - start_time,
                "best_val_mcc": best_val_mcc,
            },
            step=global_step,
        )

        print(
            f"[{task_name}] Epoch {epoch}/{config.epochs} | loss {avg_loss:.4f} | "
            f"val_mcc {val_metrics['mcc']:.4f} {'🌟 NEW BEST!' if is_best else ''}"
        )

        # Early stopping (best-val checkpoint already saved -> same reported MCC, less compute).
        if should_stop:
            print(f"[{task_name}] EARLY STOP at epoch {epoch}: no val_mcc improvement for "
                  f"{epochs_no_improve} epochs (best {best_val_mcc:.4f} @ epoch {best_epoch})")
            break

    # Final test evaluation
    test_metrics = evaluate(model, test_loader, config.device)
    wandb.log(
        {"test/f1": test_metrics["f1"], "test/mcc": test_metrics["mcc"]},
        step=global_step,
    )

    # Best test evaluation
    best_dir = os.path.join(run_dir, "best_model")
    best_test_metrics = None
    if os.path.exists(best_dir):
        best_model_state_dict = torch.load(
            os.path.join(best_dir, "student.pt"), map_location=config.device
        )
        _ = model.load_state_dict(best_model_state_dict)
        best_test_metrics = evaluate(model, test_loader, config.device)
        wandb.log(
            {
                "best_test/f1": best_test_metrics["f1"],
                "best_test/mcc": best_test_metrics["mcc"],
            },
            step=global_step,
        )

    # Save training history
    with open(os.path.join(run_dir, "training_history.json"), "w") as f:
        json.dump(training_history, f, indent=2)

    # Save final summary
    summary = {
        "task": task_name,
        "best_epoch": int(best_epoch),
        "best_val_mcc": float(best_val_mcc),
        "final_test_mcc": float(test_metrics["mcc"]),
        "final_test_f1": float(test_metrics["f1"]),
        "best_test_mcc": (
            float(best_test_metrics["mcc"]) if best_test_metrics is not None else None
        ),
        "best_test_f1": (float(best_test_metrics["f1"]) if best_test_metrics is not None else None),
        "total_epochs": int(epoch),  # ACTUAL epochs run (< config.epochs if early-stopped)
        "max_epochs": config.epochs,
        "early_stopped": bool(epoch < config.epochs),
        # Training seed — lets the 3-seed aggregator select ONLY the multi-seed runs (random_state in
        # {0,1,2}) and exclude HP-search runs (which use the config default 42), with zero contamination.
        "random_state": int(config.random_state),
        "hyperparameters": {
            "weight_ce": distillation_config.weight_ce,
            "weight_kl": distillation_config.weight_kl,
            "weight_mse": distillation_config.weight_mse,
            "temperature": distillation_config.temperature,
            "lr": config.lr,
            "batch_size": config.batch_size,
        },
    }

    with open(os.path.join(run_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also save as CSV for easy comparison
    summary_csv = os.path.join(run_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as cf:
        w = csv.writer(cf)
        w.writerow(["metric", "value"])
        w.writerow(["task", task_name])
        w.writerow(["best_epoch", best_epoch])
        w.writerow(["best_val_mcc", f"{best_val_mcc:.4f}"])
        w.writerow(["final_test_mcc", f"{test_metrics['mcc']:.4f}"])
        w.writerow(["final_test_f1", f"{test_metrics['f1']:.4f}"])
        w.writerow(
            [
                "best_test_mcc",
                (f"{best_test_metrics['mcc']:.4f}" if best_test_metrics is not None else None),
            ]
        )
        w.writerow(
            [
                "best_test_f1",
                (f"{best_test_metrics['f1']:.4f}" if best_test_metrics is not None else None),
            ]
        )

    print(f"\n{'=' * 60}")
    print(f"Training completed for {task_name}")
    print(f"Best epoch: {best_epoch} (val_mcc: {best_val_mcc:.4f})")
    print(f"Final test MCC: {test_metrics['mcc']:.4f}")
    print(f"Final test F1: {test_metrics['f1']:.4f}")
    print(
        f"Best test MCC: {best_test_metrics['mcc']:.4f}" if best_test_metrics is not None else None
    )
    print(f"Best test F1: {best_test_metrics['f1']:.4f}" if best_test_metrics is not None else None)
    print(f"Results saved to: {run_dir}")
    print(f"{'=' * 60}\n")
