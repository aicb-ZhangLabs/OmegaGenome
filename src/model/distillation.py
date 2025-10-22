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
            # Logit Standardization (CVPR 2024)
            kl = self._logit_standard_kl(s_logits, tlog)
        elif method == "dkd" and labels is not None:
            # Decoupled Knowledge Distillation (CVPR 2022)
            kl = self._dkd_loss(s_logits, tlog, labels)
        elif method == "dist":
            # DIST (NeurIPS 2022)
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
        """Logit Standardization KD Loss"""
        temp = self.config.temperature
        s_mean = s_logits.mean(dim=1, keepdim=True)
        s_std = s_logits.std(dim=1, keepdim=True) + 1e-7
        t_mean = t_logits.mean(dim=1, keepdim=True)
        t_std = t_logits.std(dim=1, keepdim=True) + 1e-7

        s_logits_norm = (s_logits - s_mean) / s_std
        t_logits_norm = (t_logits - t_mean) / t_std

        return F.kl_div(
            F.log_softmax(s_logits_norm / temp, dim=-1),
            F.softmax(t_logits_norm / temp, dim=-1),
            reduction="batchmean",
        ) * (temp**2)

    def _dkd_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor, labels: torch.Tensor):
        """Decoupled Knowledge Distillation Loss"""
        temp = self.config.temperature
        alpha = self.config.dkd_alpha
        beta = self.config.dkd_beta

        mask_target = F.one_hot(labels, num_classes=s_logits.shape[1]).bool()

        # Target class KD
        s_probs = F.softmax(s_logits / temp, dim=1)
        t_probs = F.softmax(t_logits / temp, dim=1)

        s_target = (s_probs * mask_target).sum(dim=1, keepdim=True)
        t_target = (t_probs * mask_target).sum(dim=1, keepdim=True)
        tckd_loss = F.kl_div(torch.log(s_target + 1e-8), t_target, reduction="batchmean") * (
            temp**2
        )

        # Non-target class KD
        s_logits_non_target = s_logits.masked_fill(mask_target, -1e9)
        t_logits_non_target = t_logits.masked_fill(mask_target, -1e9)
        nckd_loss = F.kl_div(
            F.log_softmax(s_logits_non_target / temp, dim=1),
            F.softmax(t_logits_non_target / temp, dim=1),
            reduction="batchmean",
        ) * (temp**2)

        return alpha * tckd_loss + beta * nckd_loss

    def _dist_loss(self, s_logits: torch.Tensor, t_logits: torch.Tensor):
        """DIST: Knowledge Distillation from A Stronger Teacher"""
        # Simplified DIST loss - correlation-based
        s_probs = F.softmax(s_logits, dim=1)
        t_probs = F.softmax(t_logits, dim=1)

        # Pearson correlation loss
        s_mean = s_probs.mean(dim=1, keepdim=True)
        t_mean = t_probs.mean(dim=1, keepdim=True)
        s_centered = s_probs - s_mean
        t_centered = t_probs - t_mean

        correlation = (s_centered * t_centered).sum(dim=1)
        s_norm = torch.sqrt((s_centered**2).sum(dim=1) + 1e-8)
        t_norm = torch.sqrt((t_centered**2).sum(dim=1) + 1e-8)

        correlation = correlation / (s_norm * t_norm)
        return (1 - correlation).mean() * self.config.temperature

    def mse_term(self, s_feats: torch.Tensor, tfeats: Optional[torch.Tensor]):
        if self.config.weight_mse > 0 and tfeats is not None:
            s_feats, tfeats = self.student_model.aligned_feats(s_feats, tfeats)
            mse = F.mse_loss(s_feats, tfeats)
            mse = self.config.weight_mse * mse
        else:
            mse = torch.tensor(0.0, device=self.device)
        return mse
