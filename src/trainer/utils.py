import os
import torch


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


@torch.no_grad()
def precompute_teacher_logits(
    tokenizer, model, sequences, batch_size, device, max_length
):
    model.eval()
    logits_list = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        tok = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = tok.input_ids.to(device)
        attention_mask = tok.attention_mask.to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits_list.append(out.logits.cpu())
    return torch.cat(logits_list, dim=0).numpy()
