"""Model 1.3/1.4: ResNet policy network (optionally Elo-conditioned). The main model.

The AlphaZero / Leela / Maia-style network from docs/cnn_resnet_tutorial.md, step 6:

    planes (N, 18, 8, 8)
      -> stem    conv 3x3 (18 -> C) + BN + ReLU                  (N, C, 8, 8)
      -> [elo]   + rating embeddings, if elo=True                (N, C, 8, 8)
      -> tower   `blocks` residual blocks                         (N, C, 8, 8)
      -> head    conv 1x1 (C -> 32) + BN + ReLU, flatten, linear  (N, 4096) move logits

The output is raw logits for all 64x64 from->to moves. Mask the illegal ones to -inf
before the softmax / cross-entropy (train.py and models.predict do this).

Knobs, all set from the YAML config (configs/*.yaml):
    blocks    depth. Start at 6, then try 10 and 12 (Maia-2 uses 12)
    channels  width. Start at 64, then try 128 and 192/256
    se        squeeze-and-excitation inside each block: the block looks at a summary of the
              whole board and re-weights its channels. Cheap; Leela Chess Zero uses it.
    elo       model 1.4: learned embeddings for the mover's and the opponent's rating bucket
              are added to the board features (the core idea of Maia-2, simplified).

Rough sizes (most parameters sit in the head's final linear layer):
    6 x 64  ~8.8M params     10 x 128  ~11.3M     12 x 256  ~22.6M
Every one of these trains comfortably on a 12 GB RTX 3060 at batch 1024 with bf16.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from chesspred.encoding import NUM_MOVES, NUM_PLANES

# Rating buckets for Elo conditioning: 100-point buckets from 400 to 2400.
# Bucket 0 means "unknown" (rating missing, stored as -1).
ELO_MIN, ELO_STEP, ELO_BUCKETS = 400, 100, 20


def elo_to_bucket(elo: torch.Tensor) -> torch.Tensor:
    """Ratings (any int dtype, -1 = unknown) -> bucket ids 0..ELO_BUCKETS as int64."""
    elo = elo.long()
    bucket = ((elo - ELO_MIN) // ELO_STEP).clamp(0, ELO_BUCKETS - 1) + 1
    return torch.where(elo < 0, torch.zeros_like(bucket), bucket)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation: average each channel over the board, then use a tiny MLP
    to decide how much of each channel to keep (a gate between 0 and 1)."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = x.mean(dim=(2, 3))                          # (N, C): one number per channel
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(gate))))
        return x * gate[:, :, None, None]


class ResidualBlock(nn.Module):
    """conv-BN-ReLU-conv-BN (+SE), add the input back (the skip), then ReLU.

     x ─────────────────────────────────────────┐  skip connection
     └─► conv ─► BN ─► ReLU ─► conv ─► BN ─► [SE] ─► (+) ─► ReLU ─► out
    """

    def __init__(self, channels: int, se: bool = False):
        super().__init__()
        # bias=False: the BatchNorm right after each conv has its own bias.
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = SqueezeExcite(channels) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        return F.relu(out + x)


class ChessResNet(nn.Module):
    """Board planes (+ optional ratings) -> 4,096 move logits."""

    def __init__(self, blocks: int = 6, channels: int = 64, head_channels: int = 32,
                 se: bool = False, elo: bool = False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.use_elo = elo
        if elo:
            # One learned vector per rating bucket, added to every square's features.
            # Zero-initialised, so adding Elo to an already-trained model starts out as a
            # no-op and can only learn to help.
            self.player_elo = nn.Embedding(ELO_BUCKETS + 1, channels)
            self.opponent_elo = nn.Embedding(ELO_BUCKETS + 1, channels)
            nn.init.zeros_(self.player_elo.weight)
            nn.init.zeros_(self.opponent_elo.weight)

        self.tower = nn.Sequential(*[ResidualBlock(channels, se) for _ in range(blocks)])

        # The 1x1 conv shrinks C channels to 32 before the big linear layer (saves params).
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(head_channels * 8 * 8, NUM_MOVES),
        )
        self._frozen: list[nn.Module] = []

    def forward(self, x: torch.Tensor, elo: torch.Tensor | None = None) -> torch.Tensor:
        """x: (N, 18, 8, 8) planes. elo: optional (N, 2) [mover, opponent] ratings, -1 = unknown.

        Models built without elo=True ignore the elo argument, so every model can be
        called the same way.
        """
        x = self.stem(x)
        if self.use_elo and elo is not None:
            buckets = elo_to_bucket(elo)
            bias = self.player_elo(buckets[:, 0]) + self.opponent_elo(buckets[:, 1])  # (N, C)
            x = x + bias[:, :, None, None].to(x.dtype)
        x = self.tower(x)
        return self.policy_head(x)

    # ---- fine-tuning helpers -----------------------------------------------------------
    def freeze_early_layers(self, n_blocks: int) -> None:
        """Freeze the stem and the first n_blocks residual blocks (roadmap week 3:
        "try freezing the early layers versus full fine-tuning"). n_blocks=0 freezes nothing."""
        self._frozen = [self.stem, *self.tower[:n_blocks]] if n_blocks > 0 else []
        for module in self._frozen:
            for p in module.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        for module in self._frozen:  # frozen BatchNorms keep their pre-trained statistics
            module.eval()
        return self


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
