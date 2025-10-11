import os
import csv
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import wandb

from dataclasses import dataclass, asdict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
from transformers import PreTrainedTokenizer
from typing import List

from .utils import precompute_teacher_logits
from ..data.dataset import (
    SeqDataset,
)


def get_best_checkpoint(parent_dir, task_name):
    task_dir = os.path.join(parent_dir, task_name)
    if not os.path.isdir(task_dir):
        return None
    ckpts = [d for d in os.listdir(task_dir) if d.startswith("checkpoint-")]
    if not ckpts:
        return None
    best = max(
        ckpts,
        key=lambda d: int(d.split("-", 1)[1]) if d.split("-", 1)[1].isdigit() else -1,
    )
    return os.path.join(task_dir, best)


def evaluate(model, loader, device):
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
    return {
        "f1": f1_score(labels, preds, average="macro"),
        "mcc": matthews_corrcoef(labels, preds),
    }


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

    print("Precomputing teacher logits...")
    train_tlogits = precompute_teacher_logits(
        teacher_tokenizer,
        teacher_model,
        X_train,
        config.batch_size,
        config.device,
        config.max_len,
    )

    train_ds = SeqDataset(X_train, y_train, config.max_len, train_tlogits)
    val_ds = SeqDataset(X_val, y_val, config.max_len)
    test_ds = SeqDataset(X_test, y_test, config.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # wandb & output dir
    prefix = f"wCE{config.weight_ce}_wKL{config.weight_kl}_wMSE{config.weight_mse}_T{config.temperature}"
    run_name = f"{task_name}_{prefix}"
    output_dir = os.path.join(config.output_dir, task_name + "_distill")
    os.makedirs(output_dir, exist_ok=True)

    wandb.init(project=config.wandb_project, name=run_name, config=asdict(config))
    wandb.watch(model, log="all", log_freq=100)

    # training
    global_step = 0
    best_val_mcc = -1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    start_time = time.time()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            ids, labs = batch[0].to(config.device), batch[1].to(config.device)
            tlog = batch[2].to(config.device) if config.weight_kl > 0 else None

            optimizer.zero_grad()
            s_logits, s_feats = model(ids, return_feats=True)

            ce = F.cross_entropy(s_logits, labs)
            loss = config.weight_ce * ce

            if config.weight_kl > 0:
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

            if config.weight_mse > 0:
                seqs_batch = [X_train[i] for i in range(len(X_train))][: len(ids)]
                tok = teacher_tokenizer(
                    seqs_batch,
                    padding="max_length",
                    truncation=True,
                    max_length=config.max_len,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    out_t = teacher_model(
                        input_ids=tok.input_ids.to(config.device),
                        attention_mask=tok.attention_mask.to(config.device),
                    )
                hidden = out_t.hidden_states[-1][:, 0, :]
                proj = model.teacher_proj(hidden)
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

        # epoch-end logging
        avg_loss = total_loss / len(train_ds)
        val_metrics = evaluate(model, val_loader, config.device)
        wandb.log(
            {
                "epoch": epoch,
                "train/epoch_loss": avg_loss,
                "val/f1": val_metrics["f1"],
                "val/mcc": val_metrics["mcc"],
                "time/elapsed_s": time.time() - start_time,
            },
            step=global_step,
        )
        print(
            f"[{task_name}] Epoch {epoch}/{config.epochs} | loss {avg_loss:.4f} | val_mcc {val_metrics['mcc']:.4f}"
        )

        # test eval every N epochs
        if epoch % config.eval_every_n_epochs == 0:
            test_ep = evaluate(model, test_loader, config.device)
            wandb.log({f"test_epoch_{epoch}_mcc": test_ep["mcc"]}, step=global_step)
            print(f"[{task_name}] Test @epoch {epoch} mcc {test_ep['mcc']:.4f}")

        # save best val
        if val_metrics["mcc"] > best_val_mcc:
            best_val_mcc = val_metrics["mcc"]
            ckpt_dir = os.path.join(
                output_dir, f"model-best-mcc_score{best_val_mcc:.4f}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "student.pt"))

    # final test
    test_metrics = evaluate(model, test_loader, config.device)
    wandb.log(
        {"test/f1": test_metrics["f1"], "test/mcc": test_metrics["mcc"]},
        step=global_step,
    )
    torch.save(model.state_dict(), os.path.join(output_dir, "model_final.pt"))

    # summary & JSON
    summary_csv = os.path.join(output_dir, "summary.csv")
    new_file = not os.path.exists(summary_csv)
    with open(summary_csv, "a", newline="") as cf:
        w = csv.writer(cf)
        if new_file:
            w.writerow(["task", "best_val_mcc", "test_mcc", "test_f1"])
        w.writerow(
            [
                task_name,
                f"{best_val_mcc:.4f}",
                f"{test_metrics['mcc']:.4f}",
                f"{test_metrics['f1']:.4f}",
            ]
        )
    with open(os.path.join(output_dir, "eval_results.json"), "w") as jf:
        json.dump(
            {
                "best_val_mcc": best_val_mcc,
                **{f"test_{k}": v for k, v in test_metrics.items()},
            },
            jf,
            indent=2,
        )
    wandb.finish()
