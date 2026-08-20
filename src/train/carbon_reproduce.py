"""Validate the Carbon install/weights by reproducing a sequence-recovery proxy.

Carbon is an autoregressive DNA model; "sequence recovery" measures how well it
predicts the next bases of a held-out continuation. Here we use the fast
teacher-forced form: feed a DNA sequence and measure next-6-mer-token accuracy
(argmax vs. the true next token). A correctly loaded Carbon-3B should score far
above the random-token baseline (~1/vocab); random weights would not.

For the exact paper metric (greedy generation, per-base accuracy over a 30-bp
continuation) run the upstream ``evaluation/sequence_recovery.py`` from
github.com/huggingface/carbon — this script is a quick environment sanity check.

    python -m src.train.carbon_reproduce --model_id HuggingFaceBio/Carbon-3B --num_seqs 64
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.dataset import DatasetConfig, build_data_splits_from_huggingface


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="HuggingFaceBio/Carbon-3B")
    ap.add_argument("--num_seqs", type=int, default=64)
    ap.add_argument("--task", default="promoter_all", help="NT task to pull human DNA from")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dna_tag", default="<dna>")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model_id} on {device} ...")
    tok = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    vocab = model.config.vocab_size
    print(f"Loaded. vocab_size={vocab}  random-token baseline ~ {1.0 / vocab:.4f}")

    # Pull a handful of real human sequences from the NT benchmark.
    X = build_data_splits_from_huggingface(DatasetConfig(task_name=args.task, data_path=""))[4]
    seqs = list(X)[: args.num_seqs]

    correct = total = 0
    with torch.no_grad():
        for seq in seqs:
            enc = tok(
                args.dna_tag + seq,
                return_tensors="pt",
                add_special_tokens=False,
                truncation=True,
                max_length=args.max_len,
            ).to(device)
            ids = enc["input_ids"]
            if ids.shape[1] < 2:
                continue
            logits = model(**enc).logits  # [1, T, V]
            preds = logits[:, :-1].argmax(-1)
            targets = ids[:, 1:]
            correct += (preds == targets).sum().item()
            total += targets.numel()

    acc = correct / max(total, 1)
    print(f"\nSequences: {len(seqs)}  tokens scored: {total}")
    print(f"Next-token (6-mer) accuracy: {acc:.4f}  (baseline {1.0 / vocab:.4f})")
    print("OK: weights load and predict non-trivially." if acc > 5.0 / vocab else
          "WARNING: accuracy near random — check tokenization / weights / dna tag.")


if __name__ == "__main__":
    main()
