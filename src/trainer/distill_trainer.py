import os
import csv
import datetime
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb
import shutil

from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import PreTrainedTokenizer
from typing import List

from .utils import precompute_teacher_logits
from ..data.dataset import (
    SeqDataset,
)


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


def create_run_directory(
    parent_dir, task_name, weight_ce, weight_kl, weight_mse, temperature, dry_run=False
):
    """
    Creates a systematic directory structure:
    {parent_dir}/{task_name}/{date}_CE{ce}_KL{kl}_MSE{mse}_T{temp}/
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hyperparam_str = f"CE{weight_ce}_KL{weight_kl}_MSE{weight_mse}_T{temperature}"
    prefix = "DRYRUN_" if dry_run else ""
    run_dir = os.path.join(
        parent_dir, task_name, f"{prefix}{timestamp}_{hyperparam_str}"
    )
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_checkpoint(model, epoch, val_mcc, run_dir, is_best=False):
    """
    Save checkpoint with systematic naming: epoch_{num}_valmcc_{score}
    Also maintains a 'best_model' directory for the best checkpoint
    """
    # Format MCC to 4 decimal places
    mcc_str = f"{val_mcc:.4f}".replace(".", "p")  # Replace . with p for filename
    epoch_dir = os.path.join(run_dir, f"epoch_{epoch}_valmcc_{mcc_str}")
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

    return epoch_dir


@dataclass
class DistillTrainerConfig:
    output_dir: str
    wandb_project: str
    epochs: int = 100
    batch_size: int = 8
    lr: float = 1e-4
    max_len: int = 1024
    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False
    log_batch_every: int = 50
    eval_every_n_epochs: int = 5
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_distill_task(
    config: DistillTrainerConfig,
    task_name: str,
    teacher_tokenizer: PreTrainedTokenizer,
    teacher_model: torch.nn.Module,
    model: torch.nn.Module,
    X_train: List[str],
    y_train: List[int],
    X_val: List[str],
    y_val: List[int],
    X_test: List[str],
    y_test: List[int],
):
    # move models to device
    model.to(config.device)
    teacher_model.to(config.device)

    needs_features = config.weight_mse > 0
    if needs_features:
        print("Precomputing teacher logits and features...")
    else:
        print("Precomputing teacher logits...")
    train_tlogits, train_tfeatures = precompute_teacher_logits(
        teacher_tokenizer,
        teacher_model,
        X_train,
        config.batch_size,
        config.device,
        config.max_len,
        needs_features=needs_features,
    )
    print("Teacher outputs precomputed.")

    train_ds = SeqDataset(
        X_train, y_train, config.max_len, train_tlogits, train_tfeatures
    )
    val_ds = SeqDataset(X_val, y_val, config.max_len)
    test_ds = SeqDataset(X_test, y_test, config.max_len)

    # Create data loaders
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size, shuffle=False, num_workers=4
    )

    # Create systematic run directory
    run_dir = create_run_directory(
        config.output_dir,
        task_name,
        config.weight_ce,
        config.weight_kl,
        config.weight_mse,
        config.temperature,
    )
    print(f"Run directory: {run_dir}")

    # Save hyperparameters
    hyperparams = asdict(config).copy()
    hyperparams["run_dir"] = run_dir
    hyperparams["timestamp"] = datetime.now().isoformat()
    # Convert any non-serializable types
    hyperparams["device"] = str(hyperparams["device"])
    with open(os.path.join(run_dir, "hyperparameters.json"), "w") as f:
        json.dump(hyperparams, f, indent=2)

    # WandB setup
    prefix = f"wCE{config.weight_ce}_wKL{config.weight_kl}_wMSE{config.weight_mse}_T{config.temperature}"
    run_name = f"{task_name}_{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project=config.wandb_project, name=run_name, config=asdict(config))
    wandb.watch(model, log="all", log_freq=100)

    best_val_mcc = -1.0
    best_epoch = 0

    global_step = 0
    start_time = time.time()

    # Training history
    training_history = []

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            ids, labs = batch[0].to(config.device), batch[1].to(config.device)
            tlog = (
                batch[2].to(config.device)
                if len(batch) > 2 and config.weight_kl > 0
                else None
            )
            tfeats = (
                batch[3].to(config.device)
                if len(batch) > 3 and config.weight_mse > 0
                else None
            )

            optimizer.zero_grad()
            s_logits, s_feats = model(ids, return_feats=True)

            ce = F.cross_entropy(s_logits, labs)
            loss = config.weight_ce * ce

            if config.weight_kl > 0 and tlog is not None:
                if config.zscore:
                    m = s_logits.mean(dim=-1, keepdims=True)
                    sd = s_logits.std(dim=-1, keepdims=True) + 1e-6
                    s_dist = (s_logits - m) / sd
                else:
                    s_dist = s_logits / config.temperature
                t_dist = tlog / config.temperature
                kl = F.kl_div(
                    F.log_softmax(s_dist, dim=-1),
                    F.softmax(t_dist, dim=-1),
                    reduction="batchmean",
                ) * (config.temperature**2)
                loss += config.weight_kl * kl
            else:
                kl = torch.tensor(0.0, device=config.device)

            if config.weight_mse > 0 and tfeats is not None:
                # Use precomputed teacher features
                proj = model.teacher_proj(tfeats)
                mse = F.mse_loss(s_feats, proj)
                loss += config.weight_mse * mse
            else:
                mse = torch.tensor(0.0, device=config.device)

            loss.backward()
            optimizer.step()

            global_step += 1
            total_loss += loss.item()

            if global_step % config.log_batch_every == 0:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/kl": kl.item(),
                        "train/mse": mse.item(),
                        "step": global_step,
                    }
                )

        # Epoch-end evaluation
        avg_loss = total_loss / len(train_loader)
        val_metrics = evaluate(model, val_loader, config.device)

        # Determine if this is the best model
        is_best = val_metrics["mcc"] > best_val_mcc
        if is_best:
            best_val_mcc = val_metrics["mcc"]
            best_epoch = epoch

        # Save checkpoint for this epoch
        checkpoint_dir = save_checkpoint(
            model, epoch, val_metrics["mcc"], run_dir, is_best=is_best
        )

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

    # Final test evaluation
    test_metrics = evaluate(model, test_loader, config.device)
    wandb.log(
        {"test/f1": test_metrics["f1"], "test/mcc": test_metrics["mcc"]},
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
        "total_epochs": config.epochs,
        "hyperparameters": {
            "weight_ce": config.weight_ce,
            "weight_kl": config.weight_kl,
            "weight_mse": config.weight_mse,
            "temperature": config.temperature,
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

    print(f"\n{'=' * 60}")
    print(f"Training completed for {task_name}")
    print(f"Best epoch: {best_epoch} (val_mcc: {best_val_mcc:.4f})")
    print(f"Final test MCC: {test_metrics['mcc']:.4f}")
    print(f"Results saved to: {run_dir}")
    print(f"{'=' * 60}\n")

    wandb.finish()
