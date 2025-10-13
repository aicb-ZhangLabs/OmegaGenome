import torch
import torch.nn.functional as F
from typing import Optional, Literal
from torch import nn
from .nn.bpnet_pytorch import BPNet
from dataclasses import dataclass


@dataclass
class BPNetClassifierConfig:
    num_labels: int = 2
    teacher_hidden_size: Optional[int] = None
    teacher_projection_opt: Literal["down", "up"] = "down"


class BPNetClassifier(nn.Module):
    def __init__(self, config: BPNetClassifierConfig):
        super().__init__()
        self.config = config
        self.bpnet = BPNet()
        C = self.bpnet.stem[0].out_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        self._define_classifier(C)

    def _define_classifier(self, C: int):
        if self.config.teacher_hidden_size is not None:
            # feature alignment is required

            if self.config.teacher_projection_opt == "down":
                self.teacher_proj = nn.Linear(self.config.teacher_hidden_size, C)
                self.classifier = nn.Linear(C, self.config.num_labels)
            elif self.config.teacher_projection_opt == "up":
                self.teacher_proj = nn.Sequential(
                    nn.Linear(C, C * 2),
                    nn.ReLU(),
                    nn.Linear(C * 2, self.config.teacher_hidden_size),
                )
                self.classifier = nn.Linear(self.config.teacher_hidden_size, self.config.num_labels)
            else:
                raise ValueError(
                    f"Invalid teacher projection option: {self.config.teacher_projection_opt}"
                )
        else:
            self.teacher_proj = None
            self.classifier = nn.Linear(C, self.config.num_labels)

    def forward(self, input_ids, return_feats: bool = False):
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = one_hot.permute(0, 2, 1)
        out = self.bpnet(x)
        feats = out["x"]
        pooled = self.pool(feats).squeeze(-1)

        # feature alignment is required
        if self.teacher_proj is not None:
            if self.config.teacher_projection_opt == "down":
                pass  # no need to do anything on student side
            elif self.config.teacher_projection_opt == "up":
                pooled = self.teacher_proj(pooled)
            else:
                raise ValueError(
                    f"Invalid teacher projection option: {self.config.teacher_projection_opt}"
                )

        logits = self.classifier(pooled)
        if return_feats:
            return logits, pooled
        return logits

    def aligned_feats(self, sfeats: torch.Tensor, tfeats: Optional[torch.Tensor] = None):
        """align the features from the student and the teacher

        :param sfeats: the features from the student
        :param tfeats: the features from the teacher
        :return: the aligned features
        """
        if self.teacher_proj is not None:
            if self.config.teacher_projection_opt == "down":
                return sfeats, self.teacher_proj(tfeats)
            elif self.config.teacher_projection_opt == "up":
                return sfeats, tfeats  # sfeats has been prejected in the forward pass
            else:
                raise ValueError(
                    f"Invalid teacher projection option: {self.config.teacher_projection_opt}"
                )
        return sfeats, tfeats
