import os
import torch

from tqdm import tqdm


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
    tokenizer, model, sequences, batch_size, device, max_length, needs_features=False
):
    model.eval()
    logits_list = []
    features_list = []

    # Debug flag to print structure once
    debug_printed = False

    for i in tqdm(range(0, len(sequences), batch_size), total=len(sequences) // batch_size):
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

            if needs_features:
                # Debug: print structure on first batch
                if not debug_printed:
                    print(f"Debug - Batch size: {len(batch)}")
                    print(f"Debug - Logits shape: {out.logits.shape}")
                    print(f"Debug - Output type: {type(out)}")
                    print(f"Debug - Has hidden_states: {hasattr(out, 'hidden_states')}")
                    if hasattr(out, "hidden_states") and out.hidden_states is not None:
                        hs = out.hidden_states
                        print(f"Debug - Hidden states type: {type(hs)}")
                        if isinstance(hs, (tuple, list)):
                            print(f"Debug - Num hidden layers: {len(hs)}")
                            print(f"Debug - Last hidden state shape: {hs[-1].shape}")
                        else:
                            print(f"Debug - Hidden states shape (single tensor): {hs.shape}")
                    debug_printed = True

                # Extract features
                if hasattr(out, "hidden_states") and out.hidden_states is not None:
                    hs = out.hidden_states

                    # Check if it's a tuple/list of tensors (standard BERT format)
                    if isinstance(hs, (tuple, list)):
                        last_hidden = hs[-1]
                    else:
                        # Single tensor
                        last_hidden = hs

                    # Now handle the dimensionality
                    if last_hidden.dim() == 3:
                        # Standard format: [batch_size, seq_len, hidden_size]
                        hidden = last_hidden[:, 0, :]  # CLS token
                    elif last_hidden.dim() == 2:
                        # DNA-BERT2 format: [seq_len, hidden_size]
                        actual_batch_size = out.logits.shape[0]
                        if actual_batch_size == 1:
                            # Single sequence: pool across sequence dimension
                            hidden = last_hidden.mean(dim=0, keepdim=True)  # [1, hidden_size]
                        else:
                            # Multiple sequences but concatenated - need to split and pool
                            seq_len_per_sample = last_hidden.shape[0] // actual_batch_size
                            hidden_list = []
                            for b in range(actual_batch_size):
                                start_idx = b * seq_len_per_sample
                                end_idx = (b + 1) * seq_len_per_sample
                                seq_hidden = last_hidden[start_idx:end_idx]
                                # Pool this sequence
                                pooled = seq_hidden.mean(dim=0)  # [hidden_size]
                                hidden_list.append(pooled)
                            hidden = torch.stack(hidden_list)  # [batch_size, hidden_size]
                    else:
                        raise ValueError(f"Unexpected hidden state shape: {last_hidden.shape}")
                else:
                    # Fallback: use pooler_output or logits
                    if hasattr(out, "pooler_output") and out.pooler_output is not None:
                        hidden = out.pooler_output
                    else:
                        print("Warning: Cannot extract hidden states, using logits as features")
                        hidden = out.logits

                features_list.append(hidden.cpu())

    logits = torch.cat(logits_list, dim=0).numpy()
    features = torch.cat(features_list, dim=0).numpy() if needs_features else None
    return logits, features
