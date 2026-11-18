"""Model architectures sharing one policy-head interface.

Every model maps   planes (N, 18, 8, 8)  [+ ratings (N, 2)]  ->  logits (N, 4096)
so train.py, evaluate.py and the app never care which architecture they are holding.

    from chesspred.models import build_model, load_model, predict
    model = build_model({"type": "resnet", "blocks": 6, "channels": 64})
    model = load_model("models/v1.pt")                       # a checkpoint saved by train.py
    predict(model, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    # -> a list like [('e5', 0.41), ('c5', 0.18), ('Nc6', 0.09)]
"""

from __future__ import annotations

from pathlib import Path

import chess
import torch
import torch.nn as nn

from chesspred.encoding import board_to_tensor, index_to_move, legal_move_mask
from chesspred.models.resnet import ChessResNet, count_params

__all__ = ["ChessResNet", "build_model", "count_params", "load_model", "predict"]


def build_model(cfg: dict) -> nn.Module:
    """Build a model from the `model:` section of a config file."""
    kind = cfg.get("type", "resnet")
    if kind == "resnet":
        return ChessResNet(
            blocks=cfg.get("blocks", 6),
            channels=cfg.get("channels", 64),
            head_channels=cfg.get("head_channels", 32),
            se=cfg.get("se", False),
            elo=cfg.get("elo", False),
        )
    # mlp.py, cnn.py and transformer.py plug in here once they're written.
    raise NotImplementedError(f"model type {kind!r} isn't implemented yet (models/{kind}.py)")


def load_model(path: str | Path, device: str | torch.device = "cpu") -> nn.Module:
    """Rebuild a model from a train.py checkpoint, ready for inference (eval mode)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)  # our own file: trusted
    model = build_model(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


@torch.no_grad()
def predict(model: nn.Module, fen: str, k: int = 3, player_elo: int | None = None,
            opponent_elo: int | None = None) -> list[tuple[str, float]]:
    """Top-k legal moves for a position as [(SAN, probability), ...]. What /predict will call."""
    model.eval()
    device = next(model.parameters()).device
    board = chess.Board(fen)
    x = board_to_tensor(board).unsqueeze(0).to(device)
    mask = legal_move_mask(board).unsqueeze(0).to(device)
    elo = torch.tensor([[player_elo if player_elo is not None else -1,
                         opponent_elo if opponent_elo is not None else -1]], device=device)
    logits = model(x, elo).float().masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, dim=1)[0]
    top = probs.topk(min(k, int(mask.sum())))
    return [(board.san(index_to_move(i.item(), board)), round(p.item(), 3))
            for p, i in zip(top.values, top.indices)]
