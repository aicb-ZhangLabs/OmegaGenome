"""
Best hyperparameters from hyperparameter search results.

This file stores the optimal hyperparameters for each combination of:
- Teacher model (nt, caduceus, enformer, dnabert2)
- Task (18 genomic tasks)
- Method (vanilla, logit_standard, dkd, dist) - for method comparison
- Model size (pico, tiny, small, etc.) - for size comparison

Update these values based on your hyperparameter search results.

Structure:
- METHOD_BEST_HYPERPARAMS[teacher][task][method] -> hyperparams dict
- SIZE_BEST_HYPERPARAMS[teacher][task][size] -> hyperparams dict
"""

from dataclasses import dataclass, asdict
from typing import Dict, Optional


@dataclass
class DistillHyperparams:
    """Hyperparameters for knowledge distillation."""

    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False
    distill_method: str = "vanilla"
    # DKD-specific
    dkd_alpha: float = 1.0
    dkd_beta: float = 8.0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# DEFAULT HYPERPARAMETERS (used when specific task/method/size not defined)
# ============================================================================

DEFAULT_VANILLA = DistillHyperparams(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,
    temperature=2.0,
    distill_method="vanilla",
)

DEFAULT_LOGIT_STANDARD = DistillHyperparams(
    weight_ce=0.1,
    weight_kl=0.9,
    weight_mse=0.0,
    temperature=2.0,
    distill_method="logit_standard",
)

DEFAULT_DKD = DistillHyperparams(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,
    temperature=4.0,
    distill_method="dkd",
    dkd_alpha=1.0,
    dkd_beta=8.0,
)

DEFAULT_DIST = DistillHyperparams(
    weight_ce=0.5,
    weight_kl=0.5,
    weight_mse=0.0,
    temperature=2.0,
    distill_method="dist",
)

DEFAULT_METHOD_HYPERPARAMS = {
    "vanilla": DEFAULT_VANILLA,
    "logit_standard": DEFAULT_LOGIT_STANDARD,
    "dkd": DEFAULT_DKD,
    "dist": DEFAULT_DIST,
}


# ============================================================================
# BEST HYPERPARAMETERS PER METHOD PER TASK
# Structure: METHOD_BEST_HYPERPARAMS[teacher][task][method] -> DistillHyperparams
#
# Update these based on your hyperparameter search results!
# ============================================================================

METHOD_BEST_HYPERPARAMS: Dict[str, Dict[str, Dict[str, DistillHyperparams]]] = {
    # =========== NT Teacher ===========
    "nt": {
        # Histone modification tasks
        "H2AFZ": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K27ac": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K27me3": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K36me3": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K4me1": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K4me2": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K4me3": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K9ac": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H3K9me3": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "H4K20me1": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        # Promoter tasks
        "promoter_all": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "promoter_tata": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "promoter_no_tata": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        # Enhancer tasks
        "enhancers": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "enhancers_types": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        # Splice site tasks
        "splice_sites_all": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,  # Updated from 2.0
                distill_method="vanilla",
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.95,  # Updated from 0.9
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.25,  # Updated from 0.5
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=2.0,  # Updated from 1.0
                dkd_beta=4.0,  # Updated from 8.0
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=1.0,  # Updated from 0.5
                temperature=1.0,  # Updated from 2.0
                distill_method="dist",
            ),
        },
        "splice_sites_acceptors": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
        "splice_sites_donors": {
            "vanilla": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "logit_standard": DistillHyperparams(
                weight_ce=0.1,
                weight_kl=0.9,
                temperature=2.0,
                distill_method="logit_standard",
            ),
            "dkd": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                temperature=4.0,
                distill_method="dkd",
                dkd_alpha=1.0,
                dkd_beta=8.0,
            ),
            "dist": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="dist"
            ),
        },
    },
    # =========== Caduceus Teacher ===========
    # Copy and modify from NT, or use defaults
    "caduceus": {},
    # =========== Enformer Teacher ===========
    "enformer": {},
    # =========== DNABERT2 Teacher ===========
    "dnabert2": {},
}


# ============================================================================
# BEST HYPERPARAMETERS PER SIZE PER TASK
# Structure: SIZE_BEST_HYPERPARAMS[teacher][task][size] -> DistillHyperparams
#
# Update these based on your hyperparameter search results!
# ============================================================================

# ============================================================
# BEST HYPERPARAMETERS PER MODEL SIZE (splice_sites_all)
# ============================================================
# Performance summary from experiments:
#  model_size    mcc distill_method  temperature  weight_ce  weight_kl  weight_mse  dkd_alpha  dkd_beta
#        pico 0.2992        vanilla       0.5000     0.5000     1.0000           1          1         8
# ultra_small 0.5942        vanilla       0.5000     0.5000     0.5000           0          1         8
# extra_small 0.8401        vanilla       4.0000     0.5000     1.0000           1          1         8
#    original 0.9078        vanilla       4.0000     0.5000     0.5000           0          1         8
# extra_large 0.8612        vanilla       1.5000     0.5000     1.0000           5          1         8  <-- BROKEN (dilation cap 6)
#       large 0.9540        vanilla       0.5000     0.5000     0.0000           1          1         8
#     xxlarge 0.9505        vanilla       0.5000     0.5000     0.0000           0          1         8
#
# NOTE: extra_large has WORSE performance than original due to dilation cap issue!
# Use extra_large_fix instead, which removes the dilation cap.

SIZE_BEST_HYPERPARAMS: Dict[str, Dict[str, Dict[str, DistillHyperparams]]] = {
    # =========== NT Teacher ===========
    "nt": {
        # Example: promoter_all task with different sizes
        "promoter_all": {
            "pico": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "ultra_tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "small": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium_small": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "original": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium_large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_large_fix": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "xxlarge": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
        },
        "H3K4me3": {
            "pico": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "ultra_tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "tiny": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "small": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium_small": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "original": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "medium_large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "extra_large_fix": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "large": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
            "xxlarge": DistillHyperparams(
                weight_ce=0.5, weight_kl=0.5, temperature=2.0, distill_method="vanilla"
            ),
        },
        "splice_sites_all": {
            # Data: pico | Temp=0.5, CE=0.5, KL=1.0, MSE=1.0
            "pico": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=1.0,
                weight_mse=1.0,
                temperature=0.5,
                distill_method="vanilla",
            ),
            # Data: ultra_small -> Key: ultra_tiny | Temp=0.5, CE=0.5, KL=0.5, MSE=0.0
            "ultra_tiny": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                weight_mse=0.0,
                temperature=0.5,
                distill_method="vanilla",
            ),
            # Data: extra_small -> Key: extra_tiny | Temp=4.0, CE=0.5, KL=1.0, MSE=1.0
            "extra_tiny": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=1.0,
                weight_mse=1.0,
                temperature=4.0,
                distill_method="vanilla",
            ),
            # Data: original | Temp=4.0, CE=0.5, KL=0.5, MSE=0.0
            "original": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.5,
                weight_mse=0.0,
                temperature=4.0,
                distill_method="vanilla",
            ),
            # Data: extra_large | Temp=1.5, CE=0.5, KL=1.0, MSE=5.0
            # NOTE: This model has poor performance due to dilation cap issue
            "extra_large": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=1.0,
                weight_mse=5.0,
                temperature=1.5,
                distill_method="vanilla",
            ),
            # NEW: extra_large_fix - Uses same hyperparams as original (which works well)
            # The architectural fix (removing dilation cap) is the key improvement
            # Start with hyperparams similar to large/xxlarge which also perform well
            "extra_large_fix": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.25,  # Same as large/xxlarge
                weight_mse=0.0,
                temperature=1.5,  # Same as large/xxlarge
                distill_method="vanilla",
            ),
            # Data: large | Temp=0.5, CE=0.5, KL=0.0, MSE=1.0
            "large": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.0,
                weight_mse=1.0,
                temperature=0.5,
                distill_method="vanilla",
            ),
            # Data: xxlarge | Temp=0.5, CE=0.5, KL=0.0, MSE=0.0
            "xxlarge": DistillHyperparams(
                weight_ce=0.5,
                weight_kl=0.0,
                weight_mse=0.0,
                temperature=0.5,
                distill_method="vanilla",
            ),
        },
        # Add more tasks as needed...
    },
    # =========== Caduceus Teacher ===========
    "caduceus": {},
    # =========== Enformer Teacher ===========
    "enformer": {},
    # =========== DNABERT2 Teacher ===========
    "dnabert2": {},
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_method_hyperparams(
    teacher: str,
    task: str,
    method: str,
) -> DistillHyperparams:
    """
    Get best hyperparameters for a specific teacher/task/method combination.

    Falls back to defaults if not found.
    """
    # Try exact match
    if teacher in METHOD_BEST_HYPERPARAMS:
        if task in METHOD_BEST_HYPERPARAMS[teacher]:
            if method in METHOD_BEST_HYPERPARAMS[teacher][task]:
                return METHOD_BEST_HYPERPARAMS[teacher][task][method]

    # Fallback to default for method
    if method in DEFAULT_METHOD_HYPERPARAMS:
        print(f"Warning: Using default hyperparams for {teacher}/{task}/{method}")
        return DEFAULT_METHOD_HYPERPARAMS[method]

    # Ultimate fallback
    print(f"Warning: Using vanilla defaults for {teacher}/{task}/{method}")
    return DEFAULT_VANILLA


def get_size_hyperparams(
    teacher: str,
    task: str,
    size: str,
) -> DistillHyperparams:
    """
    Get best hyperparameters for a specific teacher/task/size combination.

    Falls back to defaults if not found.

    Note: For extra_large_fix, if not explicitly defined, uses extra_large hyperparams
    as a starting point (the architectural fix is the main improvement).
    """
    # Try exact match
    if teacher in SIZE_BEST_HYPERPARAMS:
        if task in SIZE_BEST_HYPERPARAMS[teacher]:
            if size in SIZE_BEST_HYPERPARAMS[teacher][task]:
                return SIZE_BEST_HYPERPARAMS[teacher][task][size]
            # Fallback for extra_large_fix if not explicitly defined
            if (
                size == "extra_large_fix"
                and "extra_large" in SIZE_BEST_HYPERPARAMS[teacher][task]
            ):
                print(
                    f"Warning: Using extra_large hyperparams for extra_large_fix ({teacher}/{task})"
                )
                return SIZE_BEST_HYPERPARAMS[teacher][task]["extra_large"]

    # Fallback to vanilla default
    print(f"Warning: Using default hyperparams for {teacher}/{task}/{size}")
    return DEFAULT_VANILLA


def list_available_configs():
    """Print all available configurations."""
    print("=" * 60)
    print("METHOD BEST HYPERPARAMS")
    print("=" * 60)
    for teacher, tasks in METHOD_BEST_HYPERPARAMS.items():
        print(f"\n{teacher}:")
        for task, methods in tasks.items():
            print(f"  {task}: {list(methods.keys())}")

    print("\n" + "=" * 60)
    print("SIZE BEST HYPERPARAMS")
    print("=" * 60)
    for teacher, tasks in SIZE_BEST_HYPERPARAMS.items():
        print(f"\n{teacher}:")
        for task, sizes in tasks.items():
            print(f"  {task}: {list(sizes.keys())}")


if __name__ == "__main__":
    list_available_configs()
