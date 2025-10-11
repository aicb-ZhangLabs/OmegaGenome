import torch


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
