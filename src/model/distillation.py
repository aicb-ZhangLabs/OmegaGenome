import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple, Dict, Literal, Optional


@dataclass
class DistillationModelConfig:
    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False
    kl_method: Literal["kl", "mse"] = "kl"

    # Extended for new distillation methods
    distill_method: Literal["vanilla", "logit_standard", "dkd", "dist"] = "vanilla"
    dkd_alpha: float = 1.0  # For DKD method
    dkd_beta: float = 8.0  # For DKD method

    def __post_init__(self):
        pass


class DistillationModel(nn.Module):
    def __init__(
        self,
        config: DistillationModelConfig,
        teacher_model: nn.Module,
        student_model: nn.Module,
        device: str,
    ):
        super().__init__()
        self.config = config
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.device = device

    def prepare_batch(self, batch: List[torch.Tensor]):
        ids, labs = batch[0].to(self.device), batch[1].to(self.device)
        tlog = batch[2].to(self.device) if len(batch) > 2 and self.config.weight_kl > 0 else None
        tfeats = batch[3].to(self.device) if len(batch) > 3 and self.config.weight_mse > 0 else None
        return {
            "ids": ids,
            "labs": labs,
            "tlog": tlog,
            "tfeats": tfeats,
        }

    def get_student_knowledge(self, inputs: Dict[str, torch.Tensor]):
        """get the knowledge from the student

        :param inputs: the inputs from the prepare_batch function
        :return: the logits and features from the student
        """
        s_logits, s_feats = self.student_model(inputs["ids"], return_feats=True)
        return s_logits, s_feats

    def get_teacher_knowledge(self, inputs: Dict[str, torch.Tensor]):
        """get the knowledge from the teacher

        :param inputs: the inputs from the prepare_batch function
        :return: the logits and features from the teacher
        """
        tlog = inputs["tlog"]
        tfeats = inputs["tfeats"]
        return tlog, tfeats

    def forward(self, batch: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        inputs = self.prepare_batch(batch)
        loss, metrics = self.distillation_loss(inputs)
        return loss, metrics

    def distillation_loss(
        self, inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        s_logits, s_feats = self.get_student_knowledge(inputs)
        tlog, tfeats = self.get_teacher_knowledge(inputs)

        ce = F.cross_entropy(s_logits, inputs["labs"])
        loss = self.config.weight_ce * ce

        kl = self.kl_term(s_logits, tlog, inputs["labs"])
        loss += kl

        mse = self.mse_term(s_feats, tfeats)
        loss += mse

        return loss, {
            "loss": loss.item(),
            "ce": ce.item(),
            "kl": kl.item(),
            "mse": mse.item(),
        }

    def kl_term(
        self,
        s_logits: torch.Tensor,
        tlog: Optional[torch.Tensor],
        labels: Optional[torch.Tensor] = None,
    ):
        """Extended KL term supporting multiple distillation methods"""
        if self.config.weight_kl <= 0 or tlog is None:
            return torch.tensor(0.0, device=self.device)

        method = self.config.distill_method

        if method == "logit_standard":
            # Logit Standardization (CVPR 2024) - FIXED
            kl = self._logit_standard_kl(s_logits, tlog)
        elif method == "dkd" and labels is not None:
            # Decoupled Knowledge Distillation (CVPR 2022) - IMPROVED
            kl = self._dkd_loss(s_logits, tlog, labels)
        elif method == "dist":
            # DIST (NeurIPS 2022) - FIXED
            kl = self._dist_loss(s_logits, tlog)
        else:
            # Vanilla KD (default)
            if self.config.zscore:
                m = s_logits.mean(dim=-1, keepdims=True)
                sd = s_logits.std(dim=-1, keepdims=True) + 1e-6
                s_dist = (s_logits - m) / sd
                raise NotImplementedError("Z-scoring is not implemented")
            else:
                if self.config.kl_method == "kl":
                    s_dist = s_logits / self.config.temperature
                    t_dist = tlog / self.config.temperature
                    kl = F.kl_div(
                        F.log_softmax(s_dist, dim=-1),
                        F.softmax(t_dist, dim=-1),
                        reduction="batchmean",
                    ) * (self.config.temperature**2)
                elif self.config.kl_method == "mse":
                    kl = F.mse_loss(s_logits, tlog)
                else:
                    raise ValueError(f"Invalid KL method: {self.config.kl_method}")

        return self.config.weight_kl * kl

    def _logit_standard_kl(self, s_logits: torch.Tensor, t_logits: torch.Tensor):
        """Logit Standardization KD Loss (CVPR 2024) - CORRECTED

        Paper: "Logit Standardization in Knowledge Distillation"
        Key fix: Mean and std should be computed PER SAMPLE across classes, not per class

        Reference: https://github.com/sunshangquan/logit-standardization-KD
        """
        temp = self.config.temperature

        # FIXED: Compute mean and std PER SAMPLE (across the class dimension)
        # Shape: s_logits is (batch_size, num_classes)
        # We want to normalize each sample's logits independently
        s_mean = s_logits.mean(dim=-1, keepdim=True)  # (batch_size, 1)
        s_std = s_logits.std(dim=-1, keepdim=True, unbiased=False) + 1e-6  # (batch_size, 1)
        t_mean = t_logits.mean(dim=-1, keepdim=True)  # (batch_size, 1)
        t_std = t_logits.std(dim=-1, keepdim=True, unbiased=False) + 1e-6  # (batch_size, 1)

        # Z-score normalization per sample
        s_logits_norm = (s_logits - s_mean) / s_std
        t_logits_norm = (t_logits - t_mean) / t_std

        # Apply temperature scaling and KL divergence
        return F.kl_div(
            F.log_softmax(s_logits_norm / temp, dim=-1),
            F.softmax(t_logits_norm / temp, dim=-1),
            reduction="batchmean",
        ) * (temp**2)

    def _dkd_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor, labels: torch.Tensor):
        """Decoupled Knowledge Distillation Loss (CVPR 2022) - IMPROVED

        Paper: "Decoupled Knowledge Distillation"
        Improvement: Better consistency with official implementation

        Reference: https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
        """
        temp = self.config.temperature
        alpha = self.config.dkd_alpha
        beta = self.config.dkd_beta

        batch_size = labels.shape[0]
        gt_mask = F.one_hot(labels, num_classes=s_logits.shape[1]).bool()

        # Target class KD (TCKD) - measures "difficulty" of samples
        s_probs = F.softmax(s_logits / temp, dim=1)
        t_probs = F.softmax(t_logits / temp, dim=1)

        s_target = (s_probs * gt_mask.float()).sum(dim=1, keepdim=True)
        t_target = (t_probs * gt_mask.float()).sum(dim=1, keepdim=True)

        # Use log for numerical stability
        tckd_loss = (
            F.kl_div(torch.log(s_target + 1e-10), t_target, reduction="sum")
            * (temp**2)
            / batch_size
        )

        # Non-target class KD (NCKD) - captures dark knowledge
        # IMPROVED: Mask after temperature scaling (consistent with official implementation)
        s_probs_nckd = F.softmax(s_logits / temp - 1000.0 * gt_mask.float(), dim=1)
        t_probs_nckd = F.softmax(t_logits / temp - 1000.0 * gt_mask.float(), dim=1)

        nckd_loss = (
            F.kl_div(torch.log(s_probs_nckd + 1e-10), t_probs_nckd, reduction="sum")
            * (temp**2)
            / batch_size
        )

        return alpha * tckd_loss + beta * nckd_loss

    def _dist_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor):
        """DIST: Knowledge Distillation from A Stronger Teacher (NeurIPS 2022) - CORRECTED

        Paper: "Knowledge Distillation from A Stronger Teacher"
        Key fix: Properly implement inter-class and intra-class correlation losses

        The method computes:
        1. Inter-class correlation: Pearson correlation per class across batch dimension
        2. Intra-class correlation: Pearson correlation per instance across class dimension

        Reference: https://github.com/hunto/DIST_KD
        """

        # Get probability distributions
        s_probs = F.softmax(s_logits / self.config.temperature, dim=1)  # (B, C)
        t_probs = F.softmax(t_logits / self.config.temperature, dim=1)  # (B, C)

        # ===== Inter-class correlation =====
        # For each class, compute correlation across the batch
        # This captures how different samples relate to each class
        s_t = s_probs.t()  # (C, B)
        t_t = t_probs.t()  # (C, B)

        s_mean = s_t.mean(dim=1, keepdim=True)  # (C, 1)
        t_mean = t_t.mean(dim=1, keepdim=True)  # (C, 1)
        s_centered = s_t - s_mean  # (C, B)
        t_centered = t_t - t_mean  # (C, B)

        # Pearson correlation per class
        inter_corr = (s_centered * t_centered).sum(dim=1)  # (C,)
        s_norm = torch.sqrt((s_centered**2).sum(dim=1) + 1e-8)  # (C,)
        t_norm = torch.sqrt((t_centered**2).sum(dim=1) + 1e-8)  # (C,)
        inter_corr = inter_corr / (s_norm * t_norm + 1e-8)  # (C,)
        inter_loss = (1 - inter_corr).mean()

        # ===== Intra-class correlation =====
        # For each instance, compute correlation of its predictions across classes
        # This captures the prediction pattern for each sample
        s_mean = s_probs.mean(dim=1, keepdim=True)  # (B, 1)
        t_mean = t_probs.mean(dim=1, keepdim=True)  # (B, 1)
        s_centered = s_probs - s_mean  # (B, C)
        t_centered = t_probs - t_mean  # (B, C)

        # Pearson correlation per instance
        intra_corr = (s_centered * t_centered).sum(dim=1)  # (B,)
        s_norm = torch.sqrt((s_centered**2).sum(dim=1) + 1e-8)  # (B,)
        t_norm = torch.sqrt((t_centered**2).sum(dim=1) + 1e-8)  # (B,)
        intra_corr = intra_corr / (s_norm * t_norm + 1e-8)  # (B,)
        intra_loss = (1 - intra_corr).mean()

        # Combine both losses (paper uses equal weighting)
        return (inter_loss + intra_loss) / 2.0

    def mse_term(self, s_feats: torch.Tensor, tfeats: Optional[torch.Tensor]):
        if self.config.weight_mse > 0 and tfeats is not None:
            s_feats, tfeats = self.student_model.aligned_feats(s_feats, tfeats)
            mse = F.mse_loss(s_feats, tfeats)
            mse = self.config.weight_mse * mse
        else:
            mse = torch.tensor(0.0, device=self.device)
        return mse
