import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class DistillationModelConfig:
    weight_ce: float = 0.5
    weight_kl: float = 0.5
    weight_mse: float = 0.0
    temperature: float = 2.0
    zscore: bool = False


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
        tlog = (
            batch[2].to(self.device)
            if len(batch) > 2 and self.config.weight_kl > 0
            else None
        )
        tfeats = (
            batch[3].to(self.device)
            if len(batch) > 3 and self.config.weight_mse > 0
            else None
        )
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

    def forward(
        self, batch: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
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

        if self.config.weight_kl > 0 and tlog is not None:
            if self.config.zscore:
                m = s_logits.mean(dim=-1, keepdims=True)
                sd = s_logits.std(dim=-1, keepdims=True) + 1e-6
                s_dist = (s_logits - m) / sd
            else:
                s_dist = s_logits / self.config.temperature
                t_dist = tlog / self.config.temperature
                kl = F.kl_div(
                    F.log_softmax(s_dist, dim=-1),
                    F.softmax(t_dist, dim=-1),
                    reduction="batchmean",
                ) * (self.config.temperature**2)
                loss += self.config.weight_kl * kl
        else:
            kl = torch.tensor(0.0, device=self.device)

        if self.config.weight_mse > 0 and tfeats is not None:
            # Use precomputed teacher features
            proj = self.student_model.teacher_proj(tfeats)
            mse = F.mse_loss(s_feats, proj)
            loss += self.config.weight_mse * mse
        else:
            mse = torch.tensor(0.0, device=self.device)
        return loss, {
            "loss": loss.item(),
            "ce": ce.item(),
            "kl": kl.item(),
            "mse": mse.item(),
        }
