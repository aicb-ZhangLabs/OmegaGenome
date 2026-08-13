#!/usr/bin/env python
"""GPU extraction for the R1.13 OTHER-TEACHER panels (NT-2.5B / DNABERT-2 / Caduceus).

Companion to extract_feats_logits.py (which is Enformer-only and matches the PI's paper script
umap-tsne-260114_panel_v13_260204.py). Here we produce the same kind of cache --
SECOND_TO_LAST penultimate features + class LOGITS for {teacher, OmegaGenome student, DKD student,
from-scratch BPNet baseline} on the FULL splice_sites_all test set -- but for the HF-transformer
teachers, so feature_viz.py can render the per-teacher 2x3 panel (R1.13: "show the other teachers,
not just Enformer").

Feature definitions (paper-faithful where defined; principled analog where not):
  * BPNet student / baseline `second_to_last` == pooled feature (AdaptiveAvgPool over the stem
    output == x1.mean(dim=-1)); identical to the Enformer-panel extraction. Unchanged loaders.
  * HF teacher (NT / DNABERT-2 / Caduceus) `second_to_last` = penultimate hidden layer pooled over
    sequence positions: ``hidden_states[-2].mean(dim=1)``. The paper script only defined
    second_to_last for Enformer (= last-quarter-positions mean of the trunk embedding); for the HF
    teachers there is no PI reference, so we use the penultimate-layer mean -- one layer earlier than
    the distillation MSE feature (hidden_states[-1].mean(1)), matching the "second-to-last" name and
    the teacher-side representation the student is aligned to.

Teacher loading reuses the repo's config-driven infra:
  * NT-2.5B / DNABERT-2 -> build_glm (AutoModelForSequenceClassification, output_hidden_states=True)
  * Caduceus            -> load_caduceus_model (custom wrapper)
Teacher checkpoints resolve from code_carbon/data/finetuned_models/<teacher symlink>/ for
splice_sites_all (selection rule = the repo's find_teacher_checkpoint / get_best_checkpoint, i.e.
max mcc_score). Student + baseline ckpts are passed in (organized viz_ckpts/<teacher>/ or located
from the HP-search runs).

SLURM only (GPU). Smoke with --n 8. Saves feature_logit_data_<teacher>.npz so the Enformer cache is
never overwritten.
"""
import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import torch

REPO = "/home/pengchx3/text-dna/OmegaGenome_Revise_202606/code_carbon"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rebuttal_infra"))

from feature_space_compare import load_splice_test_seqs, NUM_LABELS, MAX_LEN  # noqa: E402
from extract_feats_logits import student_feats_logits  # noqa: E402  (BPNet student/baseline path)
from src.model.bpnet_classifier import BPNetClassifier, BPNetClassifierConfig  # noqa: E402

# Teacher hidden sizes used to size the STUDENT's teacher_proj for a clean state_dict load. These are
# the dims the distillation actually projected from (verified from the saved teacher_proj.weight
# shapes): NT-2.5B 2560, DNABERT-2 768, Caduceus 512 (= d_model 256 x2, both-strand concat). The
# teacher's own second_to_last feature for the panel is extracted independently at its real hidden dim.
# Carbon-3B's hidden dim is introspected at load time (model.config.hidden_size) rather than hardcoded,
# so TEACHER_HIDDEN has no carbon entry -- find_carbon_teacher_hidden_below resolves it dynamically.
TEACHER_HIDDEN = {"nt": 2560, "dnabert2": 768, "caduceus": 512}
# Teacher dir symlinks under code_carbon/data/finetuned_models/.
FINETUNED = os.path.join(REPO, "data/finetuned_models")
TEACHER_DIR = {
    "nt": os.path.join(FINETUNED,
        "2b5-multi-species_nucleotide-transformer-finetune-results-lora-epoch20-10-17-revised-r32-fix-num-label"),
    "dnabert2": os.path.join(FINETUNED, "dnabert2_output_shared/output"),
    "caduceus": os.path.join(FINETUNED, "caduceus_finetune_results"),
    # Carbon-3B LoRA teachers: parent dir of the 18 `{task}_finetuned/` PEFT adapters (Llama-3B base
    # + per-task LoRA + SEQ_CLS head). Same symlink the carbon distillation pipeline uses.
    "carbon": os.path.join(FINETUNED, "carbon_3b_lora"),
}


# Map a BPNet backbone feature_dim (= classifier in_features) back to its model_size string, so a
# student saved at a non-default width (e.g. the NT splice student is feature_dim 256 == "large", not
# the "original" 64) loads cleanly without hardcoding the size per teacher. Mirrors _get_hidden_dim /
# _create_bpnet_backbone in src/model/bpnet_classifier.py.
_FEATDIM_TO_SIZE = {64: "original", 7: "pico", 14: "ultra_tiny", 30: "extra_tiny", 32: "tiny",
                    90: "medium_small", 128: "medium", 120: "medium_large", 170: "extra_large",
                    256: "large", 363: "xxlarge"}


def _infer_model_size(sd):
    """Infer the BPNet model_size from a (key-normalised) state_dict by reading the classifier head's
    input feature dim. Falls back to 'original' (64) if the classifier weight isn't found."""
    for k in ("classifier.weight", "classifier.out_proj.weight"):
        if k in sd:
            feat = int(sd[k].shape[1])
            return _FEATDIM_TO_SIZE.get(feat, "original")
    return "original"


def load_bpnet_student(ckpt_path, teacher_hidden, device, from_scratch=False):
    """Load a BPNet student/baseline. `teacher_hidden` sizes the student's teacher_proj for a clean
    state_dict load (from_scratch baselines have no teacher_proj). model_size is INFERRED from the
    checkpoint's classifier feature dim (so a 256-dim "large" NT student loads without hardcoding).
    Remaps legacy `bpnet.`->`backbone.`; unused profile/total_count (and teacher_proj for from-scratch)
    heads stay uninitialised."""
    sd = torch.load(ckpt_path, map_location="cpu")
    # student.pt files may be a dict with a 'model_state_dict' / 'state_dict' wrapper.
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {(k.replace("bpnet.", "backbone.", 1) if k.startswith("bpnet.") else k): v
          for k, v in sd.items()}
    model_size = _infer_model_size(sd)
    if from_scratch:
        cfg = BPNetClassifierConfig(num_labels=NUM_LABELS, model_size=model_size)
    else:
        cfg = BPNetClassifierConfig(num_labels=NUM_LABELS, model_size=model_size,
                                    teacher_hidden_size=teacher_hidden, teacher_projection_opt="down")
    model = BPNetClassifier(cfg)
    res = model.load_state_dict(sd, strict=False)
    leftover = [k for k in res.missing_keys
                if "profile" not in k and "total_count" not in k and "teacher_proj" not in k]
    assert not leftover, f"unexpected missing keys loading {ckpt_path}: {leftover}"
    return model.to(device).eval()


def find_teacher_ckpt(teacher, task="splice_sites_all"):
    """Resolve the teacher's splice_sites_all checkpoint path/dir from its finetuned_models symlink,
    matching the repo's per-teacher layout (NT: finetuned_models/{task}_finetuned/model-best*mcc*;
    Caduceus: {task}_caduceus_finetuned/; DNABERT-2: {task}/ with pytorch_model.bin)."""
    import re
    d = TEACHER_DIR[teacher]
    if teacher == "nt":
        task_dir = os.path.join(d, "finetuned_models", f"{task}_finetuned")
        best, best_s = None, -1.0
        for x in os.listdir(task_dir):
            m = re.search(r"mcc_score([\d.]+)", x)
            if x.startswith("model-best") and m and float(m.group(1)) > best_s:
                best_s, best = float(m.group(1)), os.path.join(task_dir, x)
        return best, best_s
    if teacher == "caduceus":
        return os.path.join(d, f"{task}_caduceus_finetuned"), None
    if teacher == "dnabert2":
        return os.path.join(d, task), None
    if teacher == "carbon":
        # Carbon-LoRA layout (== get_best_checkpoint's GLM branch): {parent}/{task}_finetuned/ holding
        # the PEFT adapter (adapter_config.json). cached test MCC read from teacher_evaluation.json.
        adir = os.path.join(d, f"{task}_finetuned")
        score = None
        ev = os.path.join(adir, "teacher_evaluation.json")
        if os.path.isfile(ev):
            import json
            try:
                score = json.load(open(ev)).get("teacher_test_mcc")
            except Exception:
                pass
        return adir, score
    raise ValueError(teacher)


@torch.no_grad()
def carbon_teacher_feats_logits(ckpt, device, seqs, bs=4):
    """(logits[N,C], second_to_last feats[N,H], last feats[N,H]) for the Carbon-3B LoRA teacher.

    Reuses the carbon distillation loader verbatim: build_glm(carbon_3b_lora) merges the per-task LoRA
    adapter into the Llama-3B base + SEQ_CLS head, casts to bf16 (its fine-tune dtype), and the
    "<dna>" prefix / add_special_tokens=False tokenization is applied by tokenize_teacher_inputs --
    the exact path that produced the distilled students. We pool BOTH the penultimate layer
    (second_to_last == hidden_states[-2].mean(dim=1), the paper-method t-SNE feature) AND the last layer
    (last == hidden_states[-1].mean(dim=1), the layer the distillation MSE aligns to), so the panel can
    show either. Carbon hidden dim is introspected (model.config.hidden_size), never hardcoded."""
    from dataclasses import replace as _replace
    from src.model.glm import build_glm
    from src.trainer.utils import tokenize_teacher_inputs, _free_capture_wrappers
    from config.distillation.glm import carbon_3b_lora

    cfg = _replace(carbon_3b_lora, num_labels=NUM_LABELS, output_hidden_states=True, ckpt_path=ckpt)
    tok, model = build_glm(cfg)
    model = model.to(device).eval()
    hdim = int(model.config.hidden_size)
    print(f"[carbon] loaded teacher; hidden_size={hdim} dtype={next(model.parameters()).dtype}", flush=True)

    fo, fo_last, lo = [], [], []
    for i in range(0, len(seqs), bs):
        enc = tokenize_teacher_inputs(tok, seqs[i:i + bs], MAX_LEN,
                                      input_prefix=cfg.input_prefix, add_special_tokens=cfg.add_special_tokens)
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else torch.ones_like(ids)
        out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logits = out.logits
        hs = out.hidden_states
        second_to_last = (hs[-2] if len(hs) > 1 else hs[-1]).mean(dim=1)  # [B, H] penultimate layer
        last = hs[-1].mean(dim=1)  # [B, H] last layer (distillation MSE-aligned)
        lo.append(logits.float().cpu().numpy())
        fo.append(second_to_last.float().cpu().numpy())
        fo_last.append(last.float().cpu().numpy())
        # Carbon-3B (Llama) leaks transformers' output_hidden_states capture-wrappers across sequential
        # forwards -> GPU OOM (see _free_capture_wrappers docstring). Drop them after every batch.
        _free_capture_wrappers(model)
    return np.concatenate(lo, 0), np.concatenate(fo, 0), np.concatenate(fo_last, 0)


@torch.no_grad()
def hf_teacher_feats_logits(teacher, ckpt, device, seqs, bs=8):
    """(logits[N,C], second_to_last feats[N,H], last feats[N,H]) for an HF teacher (NT/DNABERT-2/Caduceus).
    second_to_last = hidden_states[-2].mean(dim=1) (penultimate layer, the paper-method t-SNE feature);
    last = hidden_states[-1].mean(dim=1) (last layer = the layer the distillation MSE aligns to)."""
    if teacher == "caduceus":
        from src.trainer.utils import load_caduceus_model
        import re
        best_file = None
        if os.path.isdir(ckpt):
            pat = re.compile(r"epoch(\d+)_valmcc_(-?[0-9\.]+)\.pt")
            cands = [(float(pat.match(f).group(2)), f) for f in os.listdir(ckpt) if pat.match(f)]
            if cands:
                best_file = os.path.join(ckpt, max(cands)[1])
        wrapped, tok, base = load_caduceus_model(ckpt, NUM_LABELS, device, best_file)
        model = base.eval()
    else:  # nt / dnabert2 via build_glm
        from src.model.glm import build_glm, GLMConfig
        if teacher == "nt":
            # NaN FIX (R1.13d): the NT-2.5B ESM penultimate hidden states came back all-NaN under both
            # the merged and the unmerged adapter when run in the default (effectively fp16-ish) regime
            # over max_length=1000 padded inputs -- the ESM masked-softmax over the huge pure-pad region
            # (NT 6-mers => ~100 real tokens, ~900 pad) overflows. We force fp32 (full exponent range,
            # no overflow) AND switch this teacher to dynamic `padding=True` + attention-mask-weighted
            # mean pooling (see the loop below), so the pad region is both small and excluded from the
            # pooled feature. merge_lora=False keeps the PEFT adapter live (the merge path also NaN'd).
            cfg = GLMConfig(model_name_or_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                            num_labels=NUM_LABELS, trust_remote_code=True, output_hidden_states=True,
                            torch_dtype="float32",
                            is_lora=True, base_model_path="InstaDeepAI/nucleotide-transformer-2.5b-multi-species",
                            merge_lora=False, ckpt_path=ckpt)
        else:
            cfg = GLMConfig(model_name_or_path="zhihan1996/DNABERT-2-117M", num_labels=NUM_LABELS,
                            trust_remote_code=True, output_hidden_states=True, ckpt_path=ckpt)
        tok, model = build_glm(cfg)
        if teacher == "dnabert2":
            # DNABERT-2's Triton flash-attn kernel breaks on Triton>=3 (trans_b removed). Null the
            # qkvpacked fn in its trust_remote_code module so attention falls back to pure PyTorch
            # (numerically equivalent; only the kernel differs). Must run AFTER load (modules appear
            # in sys.modules then). Reuses the repo helper from distill_nt_embedding.
            from src.train.distill_nt_embedding import _disable_dnabert2_flash_attn
            n = _disable_dnabert2_flash_attn()
            print(f"[dnabert2] Triton flash-attn disabled on {n} module(s) -> PyTorch attention", flush=True)
        model = model.to(device).eval()

    import torch.nn as nn

    def _dnabert2_pooled(ids, mask, n_drop):
        """DNABERT-2's BertModel runs on UNPADDED [total_nnz, H] states, so output_hidden_states are
        unpadded and can't be mean-pooled per sequence. Reuse the repo's padded-slice trick (see
        distill_nt_embedding.make_dnabert2_hidden_state_fn): temporarily keep the first n_layers-n_drop
        encoder blocks and call output_all_encoded_layers=False -> a clean PADDED [B,T,H] state, then
        mean over T. n_drop=0 -> last layer; n_drop=1 -> penultimate. model.bert is the underlying
        BertModel of the seq-cls wrapper."""
        bert = model.bert if hasattr(model, "bert") else model
        orig = bert.encoder.layer
        keep = max(1, len(orig) - n_drop)
        try:
            bert.encoder.layer = nn.ModuleList(list(orig)[:keep])
            seq_out, _ = bert(input_ids=ids, attention_mask=mask, output_all_encoded_layers=False)
        finally:
            bert.encoder.layer = orig
        return seq_out.mean(dim=1)  # [B, H]

    # NT uses dynamic per-batch padding (padding=True) -- padding the NT 6-mer inputs to the fixed
    # max_length=1000 left ~900 pure-pad positions whose ESM masked-softmax overflowed to NaN. Other
    # teachers keep the original fixed-max_length tokenization unchanged.
    nt_dynamic = (teacher == "nt")
    pad_kwargs = dict(padding=True) if nt_dynamic else dict(padding="max_length", max_length=MAX_LEN)
    fo, fo_last, lo = [], [], []
    for i in range(0, len(seqs), bs):
        enc = tok(seqs[i:i + bs], truncation=True, return_tensors="pt", **pad_kwargs)
        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask")
        mask = mask.to(device) if mask is not None else None
        if teacher == "dnabert2":
            logits = model(input_ids=ids, attention_mask=mask).logits
            second_to_last = _dnabert2_pooled(ids, mask, n_drop=1)
            last = _dnabert2_pooled(ids, mask, n_drop=0)
        else:
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            logits = out.logits
            hs = out.hidden_states
            penult = hs[-2] if len(hs) > 1 else hs[-1]  # [B, T, H] penultimate layer
            lasth = hs[-1]                                # [B, T, H] last layer (MSE-aligned)
            if nt_dynamic and mask is not None:
                # Attention-mask-weighted mean over real tokens only (exclude any residual pad).
                m = mask.unsqueeze(-1).to(penult.dtype)  # [B, T, 1]
                denom = m.sum(dim=1).clamp_min(1.0)
                second_to_last = (penult * m).sum(dim=1) / denom
                last = (lasth * m.to(lasth.dtype)).sum(dim=1) / denom
            else:
                second_to_last = penult.mean(dim=1)  # [B, H]
                last = lasth.mean(dim=1)             # [B, H]
        lo.append(logits.float().cpu().numpy())
        fo.append(second_to_last.float().cpu().numpy())
        fo_last.append(last.float().cpu().numpy())
    return np.concatenate(lo, 0), np.concatenate(fo, 0), np.concatenate(fo_last, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, choices=["nt", "dnabert2", "caduceus", "carbon"])
    ap.add_argument("--omega-ckpt", required=True, help="OmegaGenome (vanilla/feature-aligned) student .pt")
    ap.add_argument("--dkd-ckpt", required=True, help="DKD student .pt")
    ap.add_argument("--baseline-ckpt", required=True, help="from-scratch BPNet baseline .pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default=os.path.join(REPO, "rebuttal_infra/figs"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} teacher={args.teacher}", flush=True)

    seqs, labs = load_splice_test_seqs(args.n, os.environ.get("HF_HOME", ""))
    print(f"loaded {len(seqs)} splice_sites_all test seqs", flush=True)

    tckpt, tscore = find_teacher_ckpt(args.teacher)
    print(f"teacher ckpt={tckpt} (mcc={tscore})", flush=True)

    if args.teacher == "carbon":
        # Carbon hidden dim is introspected (not in TEACHER_HIDDEN). Extract the teacher FIRST so the
        # introspected hidden_size can size the student's teacher_proj for a clean state_dict load,
        # then free the 3B teacher before the (tiny) BPNet students so peak GPU memory stays low.
        print("loading + extracting Carbon-3B teacher second_to_last + last feats+logits...", flush=True)
        L_teacher, F_teacher, F_teacher_last = carbon_teacher_feats_logits(tckpt, device, seqs)
        th = F_teacher.shape[1]
        print(f"introspected carbon teacher_hidden={th}", flush=True)
        torch.cuda.empty_cache()
        print("loading students + from-scratch baseline...", flush=True)
        omega = load_bpnet_student(args.omega_ckpt, th, device)
        dkd = load_bpnet_student(args.dkd_ckpt, th, device)
        scratch = load_bpnet_student(args.baseline_ckpt, th, device, from_scratch=True)
        print("extracting student/baseline feats+logits...", flush=True)
        L_omega, F_omega = student_feats_logits(omega, seqs, device)
        L_dkd, F_dkd = student_feats_logits(dkd, seqs, device)
        L_base, F_baseline = student_feats_logits(scratch, seqs, device)
    else:
        th = TEACHER_HIDDEN[args.teacher]
        print("loading students + from-scratch baseline...", flush=True)
        omega = load_bpnet_student(args.omega_ckpt, th, device)
        dkd = load_bpnet_student(args.dkd_ckpt, th, device)
        scratch = load_bpnet_student(args.baseline_ckpt, th, device, from_scratch=True)

        print("extracting student/baseline feats+logits...", flush=True)
        L_omega, F_omega = student_feats_logits(omega, seqs, device)
        L_dkd, F_dkd = student_feats_logits(dkd, seqs, device)
        L_base, F_baseline = student_feats_logits(scratch, seqs, device)

        print("loading + extracting HF teacher second_to_last + last feats+logits...", flush=True)
        L_teacher, F_teacher, F_teacher_last = hf_teacher_feats_logits(args.teacher, tckpt, device, seqs)
    print(f"feat shapes: teacher={F_teacher.shape} teacher_last={F_teacher_last.shape} "
          f"omega={F_omega.shape} dkd={F_dkd.shape} base={F_baseline.shape}", flush=True)
    nan_t = np.isnan(F_teacher).any(1) | np.isnan(L_teacher).any(1)
    print(f"teacher NaN rows: {int(nan_t.sum())}/{len(nan_t)}  "
          f"F_teacher.NaN={bool(np.isnan(F_teacher).any())} "
          f"F_teacher_last.NaN={bool(np.isnan(F_teacher_last).any())} "
          f"L_teacher.NaN={bool(np.isnan(L_teacher).any())}", flush=True)
    # Strict: NO NaN allowed in the teacher features/logits before saving (the panel cosine/silhouette
    # and the saved cache must be clean). A single NaN row poisons the whole panel.
    assert not np.isnan(F_teacher).any(), "F_teacher (second_to_last) contains NaN -- fix the teacher forward before saving."
    assert not np.isnan(F_teacher_last).any(), "F_teacher_last (last layer) contains NaN -- fix the teacher forward before saving."
    assert not np.isnan(L_teacher).any(), "L_teacher contains NaN -- fix the teacher forward before saving."

    out = os.path.join(args.out, f"feature_logit_data_{args.teacher}.npz")
    np.savez_compressed(
        out,
        F_teacher=F_teacher, F_teacher_last=F_teacher_last,
        F_omega=F_omega, F_dkd=F_dkd, F_baseline=F_baseline,
        L_teacher=L_teacher, L_omega=L_omega, L_dkd=L_dkd, L_base=L_base,
        labels=np.array(labs),
    )
    print(f"SAVED {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
