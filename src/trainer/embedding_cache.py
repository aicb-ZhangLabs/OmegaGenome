"""Per-position teacher-embedding precompute + cache for the embedding-input student.

Motivation: does replacing the BPNet student's ONE-HOT input with a pretrained
DNA-model embedding help? To answer it cleanly we feed the student PER-POSITION (per-bp)
embeddings from the finetuned NT-2.5B teacher instead of one-hot, then distill exactly as
before (same teacher logits/features targets, same losses, same best-ckpt selection).

This module computes those per-bp embeddings ONCE and caches them to disk (mirroring the
existing teacher-logit/feature cache in ``utils.py``), so training runs are fast/reproducible.

6-mer -> per-bp ALIGNMENT (documented choice)
---------------------------------------------
NT tokenizes DNA as ``<cls>`` + greedy 6-mers (left to right) + single-char leftover tokens
(e.g. 1000 bp -> 1 cls + 166 six-mers + 4 single = 171 tokens). To align the teacher's
per-TOKEN hidden states to the student's per-BP position axis we:
  1. drop the ``<cls>`` (and any other special) token,
  2. EXPAND each remaining token's embedding to its character length (a 6-mer embedding is
     repeated 6x, a single-char token 1x) so every base pair inherits the embedding of the
     token that contains it,
  3. crop / zero-pad the result to ``max_len`` bp.
This is a lossless, position-faithful expansion (no interpolation), and yields a
``[max_len, hidden_size]`` per-bp vector consumed directly by ``BPNetClassifier`` in
``input_mode='nt_embedding'``.

COMPACT PER-TOKEN CACHE (memory/disk fix, semantics unchanged)
--------------------------------------------------------------
Caching the EXPANDED per-bp array ``[N, max_len, hidden]`` in float32 is ~276 GB/task
(27000 x 1000 x 2560 x 4B) and blows up / stalls when written over sshfs. Instead we cache
the COMPACT PER-TOKEN states ``[N, T<=171, hidden]`` in float16 (~24 GB/task, ~12x smaller)
plus per-token ``char_lens`` ``[N, T]`` int16, then apply the SAME ``_expand_to_perbp``
LAZILY per sample at load time (a cheap ``np.repeat``). The per-bp vectors the student sees
are byte-identical to the old path (modulo the fp16 round-trip, immaterial for KD input); the
6-mer->per-bp alignment semantics are unchanged. The cache key depends only on
(teacher_ckpt, task, split, sequence-set, max_length) -- NOT on seed/student/hyperparams --
so ONE cache per (task, split) is reused across all seeds, epochs, and the one-hot-vs-embedding
comparison.
"""

import os
import json
import hashlib
import numpy as np
import torch
from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset

from .utils import _extract_teacher_model_name


def get_embedding_cache_dir(cache_base: str, teacher_parent_dir: str, task_name: str) -> Path:
    """Cache dir for per-bp teacher embeddings: {cache_base}/data/cache_embedding/{teacher}/{task}/.

    Kept under a SEPARATE ``cache_embedding`` root (not the existing ``cache`` used for
    logits/features) so the large per-position arrays never collide with or shadow the
    pooled logit/feature cache that the KD-loss targets come from.
    """
    teacher_model_name = _extract_teacher_model_name(teacher_parent_dir)
    return Path(cache_base) / "data" / "cache_embedding" / teacher_model_name / task_name


def _embedding_cache_key(
    sequences: list, teacher_ckpt: str, max_length: int, layer_tag: str = "last"
) -> str:
    """MD5 over (n, ckpt, max_len, layer, first/last seq) — same scheme as utils._compute_cache_key.

    ``layer_tag`` (e.g. "last" or "L16") is folded into the key so embeddings from DIFFERENT
    hidden-state layers of the SAME teacher never collide in one cache file.
    """
    key_str = f"{len(sequences)}_{teacher_ckpt}_{max_length}_{layer_tag}"
    if len(sequences) > 0:
        sample = sequences[0] if len(sequences) == 1 else f"{sequences[0]}_{sequences[-1]}"
        key_str += f"_{sample}"
    return hashlib.md5(key_str.encode()).hexdigest()


def _layer_tag(embedding_layer: Optional[int]) -> str:
    """Cache-filename/key tag for a hidden-state layer selector (None->'mid', -1->'last', k->'L{k}')."""
    if embedding_layer is None:
        return "mid"
    if embedding_layer == -1:
        return "last"
    return f"L{embedding_layer}"


def embedding_cache_hit(
    cache_base: str,
    teacher_parent_dir: str,
    task_name: str,
    sequences,
    teacher_ckpt: str,
    max_length: int,
    split: str = "train",
    embedding_layer: Optional[int] = None,
) -> bool:
    """True iff a COMPLETE per-bp embedding cache for (task, split, layer) is present AND valid.

    Mirrors EXACTLY the HIT condition inside :func:`precompute_perbp_embeddings` (all three cache
    files present + metadata match on num_samples/max_length/cache_key/format), so a caller can
    decide to skip loading the (huge, sshfs-bound) embedding model when a cached read will never
    touch it. Returns False on any mismatch or read error (caller then loads the model to recompute).
    """
    layer_tag = _layer_tag(embedding_layer)
    cache_dir = get_embedding_cache_dir(cache_base, teacher_parent_dir, task_name)
    tok_path = cache_dir / f"{split}_{layer_tag}_token_embeddings.npy"
    clen_path = cache_dir / f"{split}_{layer_tag}_char_lens.npy"
    meta_path = cache_dir / f"{split}_{layer_tag}_metadata.json"
    if not (tok_path.exists() and clen_path.exists() and meta_path.exists()):
        return False
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return False
    key = _embedding_cache_key(sequences, teacher_ckpt, max_length, layer_tag)
    return (
        meta.get("num_samples") == len(sequences)
        and meta.get("max_length") == max_length
        and meta.get("cache_key") == key
        and meta.get("format") == "per_token_fp16"
    )


def _token_char_lengths(tokenizer, ids_row, special_ids: set) -> list:
    """Per-token underlying-bp length for one tokenized sequence (0 for special tokens).

    A 6-mer token -> 6, a single-char leftover token -> 1, special tokens (cls/pad/...) -> 0
    (so they contribute no bp and are dropped from the per-bp expansion).
    """
    toks = tokenizer.convert_ids_to_tokens([int(i) for i in ids_row])
    lengths = []
    for tid, tok in zip(ids_row, toks):
        if int(tid) in special_ids:
            lengths.append(0)
        else:
            # NT vocab tokens are raw nucleotide k-mers ("ACGTAC", "A", ...). len() == #bp it covers.
            lengths.append(len(tok))
    return lengths


def _expand_to_perbp(hidden, char_lens, max_len) -> np.ndarray:
    """Expand per-token hidden states [T, H] to per-bp [max_len, H] using char_lens (see module doc).

    Each token's vector is repeated ``char_lens[t]`` times along the position axis; the
    concatenation is cropped/zero-padded to ``max_len``.
    """
    # Preallocate the [max_len, H] output and fill each token's span IN PLACE. This is arithmetically
    # identical to the old "np.repeat per token + np.concatenate(pieces)" (each token's vector repeated
    # char_lens[t] times, cropped/zero-padded to max_len) but does ZERO intermediate allocations. The
    # old path allocated hundreds of tiny arrays per sample and concatenated them every __getitem__; that
    # allocation churn intermittently segfaulted numpy natively (crash at the np.concatenate). Filling a
    # single preallocated buffer removes the churn -> no crash, and it is also faster.
    H = int(hidden.shape[-1])
    out = np.zeros((max_len, H), dtype=np.float32)
    total = 0
    for t, n in enumerate(char_lens):
        if n <= 0:
            continue
        if total >= max_len:
            break
        take = n if n < max_len - total else max_len - total
        out[total : total + take] = hidden[t]  # broadcasts (H,) -> (take, H): same as np.repeat
        total += take
    return out


class PerTokenEmbeddingStore(Dataset):
    """Lazy per-bp embedding accessor backed by the COMPACT per-token fp16 cache.

    Stores per-token hidden states ``tok_emb`` ``[N, T, H]`` (fp16) and per-token bp lengths
    ``char_lens`` ``[N, T]`` (int16), and expands sample ``idx`` to the per-bp vector
    ``[max_len, H]`` (fp32) on access via :func:`_expand_to_perbp` -- the SAME expansion the
    old eager path used, so ``EmbeddingSeqDataset`` sees identical per-bp inputs while the
    on-disk/in-RAM footprint is ~12x smaller. Supports ``len()`` and integer indexing so it is
    a drop-in for the old ``np.ndarray`` returned by :func:`precompute_perbp_embeddings`
    (which ``EmbeddingSeqDataset`` indexes with ``np.ascontiguousarray(store[idx])``).

    ``tok_emb`` may be a numpy memmap (mmap_mode='r'); a single sample is small so per-item
    expansion stays cheap and never materializes the whole [N, max_len, H] array.
    """

    def __init__(self, tok_emb, char_lens, max_len: int):
        self.tok_emb = tok_emb  # [N, T, H] fp16 (possibly mmap)
        self.char_lens = char_lens  # [N, T] int
        self.max_len = max_len
        assert len(tok_emb) == len(char_lens)

    def __len__(self):
        return len(self.tok_emb)

    def __getitem__(self, idx):
        hidden = np.asarray(self.tok_emb[idx], dtype=np.float32)  # [T, H]
        clens = [int(x) for x in np.asarray(self.char_lens[idx])]
        return _expand_to_perbp(hidden, clens, self.max_len)  # [max_len, H] fp32

    @property
    def shape(self):
        return (len(self), self.max_len, int(self.tok_emb.shape[-1]))


@torch.no_grad()
def precompute_perbp_embeddings(
    tokenizer,
    teacher_model,
    sequences,
    batch_size: int,
    device: str,
    max_length: int,
    *,
    cache_base: str,
    teacher_parent_dir: str,
    task_name: str,
    teacher_ckpt: str,
    use_cache: bool = True,
    split: str = "train",
    embedding_layer: Optional[int] = None,
    hidden_state_fn=None,
    load_in_ram: bool = True,
) -> "PerTokenEmbeddingStore":
    """Return a per-bp embedding accessor for ``sequences``, backed by a compact cache.

    Caches the COMPACT PER-TOKEN states ``[N, T<=171, H]`` (fp16) + per-token bp lengths
    ``char_lens`` ``[N, T]`` (int16), NOT the 12x-larger expanded per-bp array, then wraps them
    in :class:`PerTokenEmbeddingStore` which expands to per-bp ``[max_length, H]`` (fp32) lazily
    per sample via :func:`_expand_to_perbp`. The student therefore sees the SAME per-bp inputs as
    before (the 6-mer->per-bp alignment is unchanged), while the cache is small enough to persist
    reliably (the old ~276 GB/task per-bp fp32 write stalled/failed over sshfs).

    Cache is keyed on (n / teacher_ckpt / max_len / first+last seq) only -- independent of seed,
    student and hyperparams -- so it is a HIT on any subsequent run (all epochs/seeds and the
    one-hot-vs-embedding comparison reuse one cache per (task, split)). Loads via ``mmap_mode='r'``
    so the whole array never lands in RAM.

    ``split`` distinguishes the train/val/test cache files (the student consumes all three).

    ``embedding_layer`` selects WHICH teacher hidden-state layer feeds the student:
      * ``None`` (default) -> the MIDDLE layer ``n_hidden_states // 2`` (more transferable than the
        task-specialized last layer of a fine-tuned teacher; ~layer 16 of NT-2.5B's 33 states),
      * a negative index (e.g. ``-1`` for the last layer) or a non-negative index selects that
        ``out.hidden_states`` entry directly.
    The chosen layer is folded into BOTH the cache filenames and the cache key, so embeddings from
    different layers of the same teacher never collide.

    ``hidden_state_fn`` lets a teacher with a NON-standard forward provide its per-token layer states.
    Signature ``fn(teacher_model, input_ids, attention_mask, embedding_layer) -> ([B, T, H] tensor,
    resolved_layer_index)``. Default (None) uses the standard HF path ``out.hidden_states`` (NT,
    Caduceus, ...): a tuple of ``n_layers+1`` states, ``embedding_layer=None`` -> middle ``n//2``.
    DNABERT-2's ``BertModel`` returns ``(encoded_layers, pooled)`` (no ``hidden_states`` attr) and its
    all-layers path is UNPADDED, so the driver passes a ``hidden_state_fn`` that slices the encoder to
    the first L layers and reads the clean padded ``sequence_output`` (see distill_nt_embedding.py).

    ``load_in_ram`` (default True): on a cache HIT, read the compact fp16 cache FULLY into RAM instead
    of memmapping it. The galaxy SSD is an SSHFS mount on laniakea/voyager and a memmap READ over it
    hangs the training loop on any sshfs blip (this is exactly what hung the NT token-embedding jobs at step 0).
    The compact cache is small enough (~tens of GB max; nodes have 48-64GB) to hold in RAM. Set False
    only on a node-local (non-sshfs) cache where mmap is safe and RAM is tight.
    """
    # Layer tag for cache filenames/key. "last"==-1, "mid"==None (resolved per-model at forward time),
    # else "L{idx}". Resolution of the actual index happens once the first forward exposes
    # len(out.hidden_states); the on-disk metadata records the resolved index for auditability.
    layer_tag = _layer_tag(embedding_layer)

    cache_dir = get_embedding_cache_dir(cache_base, teacher_parent_dir, task_name)
    tok_path = cache_dir / f"{split}_{layer_tag}_token_embeddings.npy"
    clen_path = cache_dir / f"{split}_{layer_tag}_char_lens.npy"
    meta_path = cache_dir / f"{split}_{layer_tag}_metadata.json"
    key = _embedding_cache_key(sequences, teacher_ckpt, max_length, layer_tag)

    if use_cache and tok_path.exists() and clen_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if (
                meta.get("num_samples") == len(sequences)
                and meta.get("max_length") == max_length
                and meta.get("cache_key") == key
                and meta.get("format") == "per_token_fp16"
            ):
                # CRITICAL (sshfs hang fix): the galaxy SSD is an SSHFS mount on laniakea/voyager, and
                # a numpy MEMMAP into a cache file on it hangs the training loop on any sshfs blip
                # (precompute's sequential WRITE survives; the per-step lazy memmap READ at training
                # start does not — this is what hung the 10 NT token-embedding jobs at step 0). The compact fp16
                # per-token cache is small (~tens of GB max), so by default load it FULLY INTO RAM
                # (no mmap_mode) so every __getitem__ reads from memory, never the sshfs mount.
                mmap = None if load_in_ram else "r"
                kind = "in-RAM" if load_in_ram else "mmap"
                print(f"[emb-cache HIT] {split} ({kind}): {tok_path}")
                return PerTokenEmbeddingStore(
                    np.load(tok_path, mmap_mode=mmap),
                    np.load(clen_path, mmap_mode=mmap),
                    max_length,
                )
        except Exception as e:
            print(f"[emb-cache] failed to read {cache_dir}: {e}; recomputing")

    print(
        f"[emb-cache MISS] computing per-token embeddings for {task_name}/{split} ({len(sequences)} seqs)..."
    )
    teacher_model.eval()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    tok_rows = []  # list of [T_i, H] fp16
    clen_rows = []  # list of [T_i] int
    resolved_layer = None  # filled on first batch (depends on len(out.hidden_states))

    for i in range(0, len(sequences), batch_size):
        batch_seq = list(sequences[i : i + batch_size])
        enc = tokenizer(
            batch_seq,
            padding="longest",
            truncation=True,
            max_length=max_length,  # token cap (>= bp/6 + specials); per-bp crop happens at expand
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        if hidden_state_fn is not None:
            # Teacher-specific forward (e.g. DNABERT-2 layer-slice); returns ([B,T,H], resolved_idx).
            per_token, rl = hidden_state_fn(
                teacher_model, input_ids, attention_mask, embedding_layer
            )
            if resolved_layer is None:
                resolved_layer = rl
                print(
                    f"[emb-cache] custom hidden_state_fn; using layer index {resolved_layer} (tag '{layer_tag}')"
                )
        else:
            out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            # Standard HF path. out.hidden_states is a tuple of length (n_layers + 1)
            # [embeddings, layer1, ..., layerN]. embedding_layer=None -> MIDDLE layer (n // 2) which
            # transfers better than the fine-tuned last layer; -1 -> last; else the index.
            hidden_states = out.hidden_states
            n_hs = len(hidden_states)
            if resolved_layer is None:
                resolved_layer = (n_hs // 2) if embedding_layer is None else embedding_layer
                print(
                    f"[emb-cache] teacher exposes {n_hs} hidden states; "
                    f"using layer index {resolved_layer} (tag '{layer_tag}')"
                )
            per_token = hidden_states[resolved_layer]
        # Cache per-TOKEN states in fp16 (compact); per-bp expansion happens lazily on load.
        hs = per_token.half().cpu().numpy()  # [B, T, H] fp16
        ids_cpu = enc.input_ids.cpu().numpy()
        for b in range(hs.shape[0]):
            char_lens = _token_char_lengths(tokenizer, ids_cpu[b], special_ids)
            tok_rows.append(hs[b])
            clen_rows.append(np.asarray(char_lens, dtype=np.int16))
        del per_token, hs, input_ids, attention_mask

    # Pad ragged per-batch token lengths to a common T_max (pad tokens get char_len 0, which
    # _expand_to_perbp skips, so padding is inert and the per-bp result is unchanged).
    H = tok_rows[0].shape[-1]
    t_max = max(r.shape[0] for r in tok_rows)
    n = len(tok_rows)
    tok_emb = np.zeros((n, t_max, H), dtype=np.float16)
    char_lens = np.zeros((n, t_max), dtype=np.int16)
    for j, (te, cl) in enumerate(zip(tok_rows, clen_rows)):
        tok_emb[j, : te.shape[0]] = te
        char_lens[j, : cl.shape[0]] = cl

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

        def _atomic_save(path: Path, arr):
            tmp = str(path) + f".tmp.{os.getpid()}"
            np.save(tmp, arr)  # np.save appends .npy
            os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, path)

        _atomic_save(tok_path, tok_emb)
        _atomic_save(clen_path, char_lens)
        meta = {
            "num_samples": len(sequences),
            "max_length": max_length,
            "hidden_size": int(H),
            "token_len": int(t_max),
            "teacher_checkpoint": teacher_ckpt,
            "cache_key": key,
            "split": split,
            "format": "per_token_fp16",
            "layer_tag": layer_tag,
            "resolved_layer": int(resolved_layer) if resolved_layer is not None else None,
        }
        tmp_m = str(meta_path) + f".tmp.{os.getpid()}"
        Path(tmp_m).write_text(json.dumps(meta, indent=2))
        os.replace(tmp_m, meta_path)
        gb = tok_emb.nbytes / 1e9
        print(f"[emb-cache] saved per-token {tok_emb.shape} fp16 ({gb:.1f} GB) -> {tok_path}")

    return PerTokenEmbeddingStore(tok_emb, char_lens, max_length)
