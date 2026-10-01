"""THE ONE PLACE to edit for machine-specific paths in the Carbon distillation pipeline.

To reproduce on a different machine, either (a) set the environment variables below, or (b) edit the
defaults here — nothing else in the distillation code hardcodes a machine path. Defaults target the
original lab server (galaxy SSD + /home).

  CARBON_TEACHER_DIR   parent dir of the 18 `{task}_finetuned/` LoRA adapters. Read once per task at
                       precompute, so a node-agnostic location (e.g. ~/...) is fine. A fresh machine
                       can pull these from HF: `explcre/carbon-3b-lora-teachers-nt18`.
  CARBON_OUTPUT_BASE   base dir for student checkpoints / distillation outputs. Prefer a fast local
                       disk; the default is $OG_SCRATCH if it exists, else $HOME.
"""

import os


def _default_output_base() -> str:
    """Fast-disk base for outputs: $OG_SCRATCH if it exists, else the user's home."""
    base = os.environ.get("OG_SCRATCH", "")
    if base and os.path.isdir(base):
        return base
    return os.path.expanduser("~")


CARBON_TEACHER_DIR = os.environ.get(
    "CARBON_TEACHER_DIR",
    os.path.expanduser("~/carbon_teachers/carbon_3b_lora"),
)

CARBON_OUTPUT_BASE = os.environ.get("CARBON_OUTPUT_BASE", _default_output_base())

# HF repo mirroring the teachers (for download on a fresh machine).
CARBON_TEACHER_HF_REPO = os.environ.get(
    "CARBON_TEACHER_HF_REPO", "explcre/carbon-3b-lora-teachers-nt18"
)
