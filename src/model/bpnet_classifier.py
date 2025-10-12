import torch.nn.functional as F
from typing import Optional
from torch import nn
from .nn.bpnet_pytorch import BPNet
from dataclasses import dataclass


@dataclass
class BPNetClassifierConfig:
    num_labels: int = 2
    teacher_hidden_size: Optional[int] = None


class BPNetClassifier(nn.Module):
    def __init__(self, config: BPNetClassifierConfig):
        super().__init__()
        self.config = config
        self.bpnet = BPNet()
        C = self.bpnet.stem[0].out_channels
        self.pool = nn.AdaptiveAvgPool1d(1)
        if config.teacher_hidden_size is not None:
            self.teacher_proj = nn.Sequential(
                nn.Linear(C, C * 2),
                nn.GELU(),
                nn.Linear(C * 2, config.teacher_hidden_size),
            )
            self.classifier = nn.Linear(config.teacher_hidden_size, config.num_labels)
        else:
            self.teacher_proj = None
            self.classifier = nn.Linear(C, config.num_labels)

    def forward(self, input_ids, return_feats: bool = False):
        one_hot = F.one_hot(input_ids, num_classes=4).float()
        x = one_hot.permute(0, 2, 1)
        out = self.bpnet(x)
        feats = out["x"]
        pooled = self.pool(feats).squeeze(-1)
        if self.teacher_proj is not None:
            pooled = self.teacher_proj(pooled)
        logits = self.classifier(pooled)
        if return_feats:
            return logits, pooled
        return logits
