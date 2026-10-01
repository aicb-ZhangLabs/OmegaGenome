import torch
import torch.nn as nn

from einops.layers.torch import Reduce, Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class BPNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(4, 64, 25, padding="same"),
            nn.ReLU(),
            *[
                Residual(
                    nn.Sequential(nn.Conv1d(64, 64, 3, padding="same", dilation=2**i), nn.ReLU())
                )
                for i in range(1, 10)
            ],
        )

        self.profile_head = nn.Sequential(
            nn.ConvTranspose1d(64, 2, 25, padding=12),  # padding=12 works like padding='same' here.
            Rearrange("b c l -> b l c"),
        )

        self.total_count_head = nn.Sequential(
            Reduce("b c l -> b c", "mean"),  # Global average pooling.
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.stem(x)

        return {
            "x": x,
            "profile": self.profile_head(x),
            "total_count": self.total_count_head(x),
        }


class BPNetWidth(BPNet):
    """Width-parametrized BPNet (param-match): IDENTICAL architecture to BPNet but with a
    configurable channel width ``C`` (BPNet hard-codes C=64).

    Why a *subclass*: the embedding front-ends in BPNetClassifier (latefuse_onehot / replaceK) gate on
    ``isinstance(self.backbone, BPNet)`` and reach into ``self.stem`` / ``self.profile_head`` etc. A
    subclass passes that isinstance check, so the matched-size student supports the SAME arms as the
    original-size student (no special-casing needed downstream). Structure is byte-for-byte the original:
    a Conv1d(4,C,25) + ReLU stem, 9 dilated (uncapped 2**i, i=1..9) Residual blocks, plus the profile /
    total_count heads (kept so the deployable-backbone param accounting matches `original`).

    C=61 makes backbone+classifier = 110,477; with the replace4 (D->4) embedding adapter (10,244) the
    deployable total = 120,721 ~= the one-hot baseline 121,094 (diff -373), removing the embedding
    student's capacity advantage so any accuracy delta is purely the input representation.
    """

    def __init__(self, channels: int = 61):
        nn.Module.__init__(
            self
        )  # bypass BPNet.__init__ (which hard-codes C=64); rebuild at width C
        C = channels
        self.channels = C
        self.stem = nn.Sequential(
            nn.Conv1d(4, C, 25, padding="same"),
            nn.ReLU(),
            *[
                Residual(
                    nn.Sequential(nn.Conv1d(C, C, 3, padding="same", dilation=2**i), nn.ReLU())
                )
                for i in range(1, 10)
            ],
        )
        self.profile_head = nn.Sequential(
            nn.ConvTranspose1d(C, 2, 25, padding=12),
            Rearrange("b c l -> b l c"),
        )
        self.total_count_head = nn.Sequential(
            Reduce("b c l -> b c", "mean"),
            nn.Linear(C, 2),
        )


if __name__ == "__main__":
    model = BPNet()

    x = torch.randn(1, 4, 5000, requires_grad=True)
    out = model(x)

    print(out["x"].shape)  # (1, 64, 5000)
    print(out["profile"].shape)  # (1, 5000, 2)
    print(out["total_count"].shape)  # (1, 2)

    loss = out["x"][:, 0, 2500].sum()
    loss.backward()

    print((x.grad != 0.0).sum() / 4)  # (1, 4, 3000)
