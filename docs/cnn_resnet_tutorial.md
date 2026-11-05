# Tutorial: A Simple CNN and a ResNet for Chess Move Prediction

> **Last updated:** 2026-11-05
> **Companion docs:** [`roadmap.md`](./roadmap.md) (this is roadmap weeks 2 and 3, models 1.2 and 1.3) · [`resources.md`](./resources.md)
> **Time:** about 2–3 hours to work through · **Tested:** every code block was run end to end on your `data/chess_com_games_2026-11-04.pgn` (PyTorch 2.x, python-chess 1.11)

---

## What you'll build

Two PyTorch models that take a **chess position** and output a **probability for every possible move**:

1. **SimpleCNN**: three convolution layers followed by one big linear layer. Easy to understand.
2. **ChessResNet**: the AlphaZero/Maia-style residual network. This is the architecture your project builds on.

Both use the design from the roadmap (decisions #4 and #5):

```
 python-chess Board
        │
        ▼
 board_to_tensor()  ──►  18 × 8 × 8 tensor   (pieces, side to move, castling, en passant)
        │
        ▼
     the model      ──►  4,096 logits         (one per from-square × to-square pair)
        │
        ▼
 legal-move mask    ──►  illegal moves set to −∞
        │
        ▼
     softmax        ──►  probability of each legal move  →  top-1 / top-3 prediction
```

Everything goes in **one Python file** (for example `src/chesspred/tutorial.py`). Add each step's code block to the file in order. Steps 1–8 define the pieces, and step 9 runs them.

---

## Step 0: Setup

```bash
pip install chess torch numpy
```

On the RTX 3060 machine, install the CUDA build of PyTorch using the selector at [pytorch.org](https://pytorch.org/get-started/locally/). Colab already has it.

Start your file with the imports:

```python
import random
import sys

import chess
import chess.pgn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
```

---

## Step 1: Turn a board into a tensor

A CNN needs numbers in a grid. The board is described with **18 "planes"**, each an 8×8 grid of 0s and 1s:

| Planes | Meaning                                                                           |
| ------ | --------------------------------------------------------------------------------- |
| 0–5    | **Our** pawns, knights, bishops, rooks, queens, king (1 where that piece is)      |
| 6–11   | **Their** pawns, knights, bishops, rooks, queens, king                            |
| 12     | All 1s if we are White, all 0s if we are Black                                    |
| 13–16  | Castling rights: ours kingside, ours queenside, theirs kingside, theirs queenside |
| 17     | The en passant target square (if an en passant capture is legal)                  |

**The key trick is to always look from the side-to-move's point of view.** If it's Black's turn, flip the board with `board.mirror()`. That flips the ranks and swaps the colors, so the player to move is always "White moving up the board". The model then only has to learn chess once, not separately for White and Black.

```python
NUM_PLANES = 18
NUM_MOVES = 64 * 64  # from-square x to-square


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode a board as an 18x8x8 tensor, always from the side-to-move's point of view."""
    white_to_move = board.turn == chess.WHITE
    if not white_to_move:
        board = board.mirror()  # flip ranks + swap colors, so "us" is always White

    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    # Planes 0-5: our pieces (P,N,B,R,Q,K). Planes 6-11: their pieces.
    for square, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        row, col = divmod(square, 8)  # square = rank * 8 + file
        x[plane, row, col] = 1.0

    # Plane 12: were we originally White? (keeps first-move info after mirroring)
    x[12, :, :] = 1.0 if white_to_move else 0.0

    # Planes 13-16: castling rights (ours K-side, ours Q-side, theirs K-side, theirs Q-side)
    x[13, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    x[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    x[15, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    x[16, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # Plane 17: en passant target square (only if an en passant capture is actually legal)
    if board.has_legal_en_passant():
        row, col = divmod(board.ep_square, 8)
        x[17, row, col] = 1.0

    return torch.from_numpy(x)
```

**Try it** (in a Python shell or notebook):

```python
b = chess.Board()
t = board_to_tensor(b)
print(t.shape)        # torch.Size([18, 8, 8])
print(t[0])           # plane 0 = our pawns: a row of 1s on row 1 (the 2nd rank)
```

> **Why `row, col = divmod(square, 8)`?** python-chess numbers squares 0–63: a1 = 0, b1 = 1, …, h1 = 7, a2 = 8, …, h8 = 63. So `square // 8` is the rank (row) and `square % 8` is the file (column).

---

## Step 2: Turn a move into a class number (and back)

The model is a **classifier**. Each possible move is a class. A move is "from square → to square", and there are 64 × 64 = **4,096** combinations, so:

```
index = from_square * 64 + to_square
```

Most of those 4,096 are never legal (like a1 → a1), but that's fine because they get masked out (step 3).

The move must be **mirrored the same way the board was** when it's Black's turn.

```python
def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Map a move to one of 4096 classes (from * 64 + to), in the mover's point of view."""
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index for the given board (promotions become queen promotions)."""
    from_sq, to_sq = divmod(index, 64)
    if board.turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    move = chess.Move(from_sq, to_sq)
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        move.promotion = chess.QUEEN
    return move
```

> **Promotions:** e7→e8=Q and e7→e8=N share the same from→to index. To keep things simple, every promotion is predicted as a queen. Underpromotions are rare enough to ignore for now.

---

## Step 3: The legal-move mask

The model outputs a score for all 4,096 moves, but only about 30 are legal in a typical position. A **mask** marks which ones are legal, so the illegal ones can be removed before turning scores into probabilities.

```python
def legal_move_mask(board: chess.Board) -> torch.Tensor:
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board.turn)] = True
    return mask
```

This is also what makes the **random baseline** easy to compute: picking uniformly among legal moves gets about 1/30 right.

---

## Step 4: Load positions from a PGN

Each **training example** is one position plus the move that was actually played:

```python
def load_positions(pgn_path, player=None):
    """Return a list of (game_id, fen, move_uci). If player is set, keep only their moves."""
    samples = []
    with open(pgn_path) as f:
        game_id = 0
        while (game := chess.pgn.read_game(f)) is not None:
            board = game.board()
            for move in game.mainline_moves():
                mover = game.headers["White"] if board.turn == chess.WHITE else game.headers["Black"]
                if player is None or mover.lower() == player.lower():
                    samples.append((game_id, board.fen(), move.uci()))
                board.push(move)
            game_id += 1
    return samples
```

- `player=None` keeps **every** move (yours and your opponents'). That's more data, which is good for learning chess in general.
- `player="whopizza1"` keeps **only your moves**. Use this for **fine-tuning** and the **headline evaluation** (roadmap week 3).

A PyTorch `Dataset` turns each sample into `(board tensor, move index, legal mask)`:

```python
class ChessMoveDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        _, fen, uci = self.samples[i]
        board = chess.Board(fen)
        x = board_to_tensor(board)
        y = move_to_index(chess.Move.from_uci(uci), board.turn)
        mask = legal_move_mask(board)
        return x, y, mask
```

> Storing FEN strings and encoding them inside `__getitem__` is simple, but slow for millions of positions. In roadmap week 2 you'll **pre-encode** everything into arrays once (see "Moving to the RTX 3060" below).

---

## Step 5: Model A, the SimpleCNN

```python
class SimpleCNN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.policy = nn.Linear(channels * 8 * 8, NUM_MOVES)

    def forward(self, x):          # x: (batch, 18, 8, 8)
        x = self.features(x)       # -> (batch, 64, 8, 8)
        x = x.flatten(1)           # -> (batch, 4096)
        return self.policy(x)      # -> (batch, 4096) move logits
```

**What each part does**

| Layer | Output shape | What it learns |
|---|---|---|
| Input | 18 × 8 × 8 | — |
| Conv 3×3 (18→64) + ReLU | 64 × 8 × 8 | Local patterns: "pawn next to king", "knight attacks this square" |
| Conv 3×3 (64→64) + ReLU | 64 × 8 × 8 | Patterns of patterns, now seeing a 5×5 area |
| Conv 3×3 (64→64) + ReLU | 64 × 8 × 8 | Now a 7×7 area: almost the whole board |
| Flatten | 4,096 | — |
| Linear (4,096→4,096) | 4,096 | Turns board features into a score for each move |

- `padding=1` keeps the board 8×8 after each conv.
- The output is **logits** (raw scores), not probabilities. Softmax comes later.
- **Parameters: ~16.9M**, and almost all of them (16.8M) are in that last linear layer. That's a weakness the ResNet fixes.

---

## Step 6: Model B, the ResNet

### What's a residual block?

A normal layer computes `output = f(x)`. A **residual block** computes `output = f(x) + x`: it learns a *correction* on top of its input.

```
 x ──────────────────────────────┐   (skip connection)
 │                               │
 └─► conv ─► BN ─► ReLU ─► conv ─► BN ─► (+) ─► ReLU ─► output
```

Why it matters:
- You can stack **many** blocks (Maia-2 uses 12) without training breaking down, because the skip connection lets gradients flow straight back.
- **BatchNorm (BN)** keeps the values in a stable range, which makes training faster and less sensitive to the learning rate.

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        skip = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + skip)    # the "residual" connection
```

(`bias=False` is used because BatchNorm right after the conv already has its own bias.)

### The full network

```python
class ChessResNet(nn.Module):
    def __init__(self, blocks=6, channels=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_MOVES),
        )

    def forward(self, x):          # x: (batch, 18, 8, 8)
        x = self.stem(x)           # -> (batch, 64, 8, 8)
        x = self.tower(x)          # -> (batch, 64, 8, 8)
        return self.policy_head(x) # -> (batch, 4096)
```

| Part | Output shape | Notes |
|---|---|---|
| **Stem**: conv 3×3 (18→64) + BN + ReLU | 64 × 8 × 8 | Lifts the 18 planes into 64 feature channels |
| **Tower**: 6 residual blocks | 64 × 8 × 8 | 12 conv layers deep. **This is where the "chess understanding" happens** |
| **Policy head**: conv 1×1 (64→32) + BN + ReLU | 32 × 8 × 8 | Shrinks the channels before the big linear layer, which saves parameters |
| Flatten + Linear (2,048→4,096) | 4,096 | One logit per move |

**Parameters: ~8.8M.** That's about half the SimpleCNN's, but it's **much deeper**. The 1×1 conv in the head is what keeps it small.

**Knobs to turn later:** `blocks` (depth) and `channels` (width). The roadmap suggests starting at 6 × 64 and scaling up while validation accuracy improves.

---

## Step 7: Training

```python
def masked(logits, mask):
    return logits.masked_fill(~mask, float("-inf"))


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = masked(model(x), mask)
        loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y)
        n += len(y)
    return total_loss / n
```

**How it works**
- `masked()` sets illegal moves to −∞. After softmax they get probability **exactly 0**, so the model never wastes effort learning that a1→a1 is a bad idea.
- `F.cross_entropy` = softmax + "how surprised was the model by the real move?" Lower is better.
- The 5 lines inside the loop are the standard PyTorch training step: forward, loss, zero the gradients, backward, step.

---

## Step 8: Evaluate and predict

```python
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    top1 = top3 = n = 0
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = masked(model(x), mask)
        best3 = logits.topk(3, dim=1).indices          # (batch, 3)
        top1 += (best3[:, 0] == y).sum().item()
        top3 += (best3 == y.unsqueeze(1)).any(dim=1).sum().item()
        n += len(y)
    return top1 / n, top3 / n


@torch.no_grad()
def predict(model, fen, k=3, device="cpu"):
    model.eval()
    board = chess.Board(fen)
    x = board_to_tensor(board).unsqueeze(0).to(device)
    mask = legal_move_mask(board).unsqueeze(0).to(device)
    probs = torch.softmax(masked(model(x), mask), dim=1)[0]
    top = probs.topk(k)
    return [(board.san(index_to_move(i.item(), board)), round(p.item(), 3))
            for p, i in zip(top.values, top.indices)]
```

- `model.eval()` matters for the ResNet: it makes BatchNorm use its saved statistics instead of the current batch's.
- `predict()` is basically what the app's `/predict` endpoint will call in roadmap week 4.

---

## Step 9: Put it all together

Add these helpers and a `main()`:

```python
def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_device():
    if torch.cuda.is_available():
        return "cuda"          # your RTX 3060 / Colab
    if torch.backends.mps.is_available():
        return "mps"           # Apple Silicon
    return "cpu"


def roundtrip_test(samples):
    """Every real move must be legal in the mask and decode back to itself."""
    for _, fen, uci in samples:
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci)
        idx = move_to_index(move, board.turn)
        assert legal_move_mask(board)[idx], (fen, uci)
        back = index_to_move(idx, board)
        if move.promotion in (None, chess.QUEEN):
            assert back == move, (fen, uci, back)
    print(f"Round-trip test passed on {len(samples)} moves")


def main(pgn_path):
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    print("device:", device)

    samples = load_positions(pgn_path)
    roundtrip_test(samples)

    # Split by GAME, not by position (positions from one game are too similar).
    game_ids = sorted({g for g, _, _ in samples})
    cut = int(len(game_ids) * 0.8)
    train_games = set(game_ids[:cut])
    train = [s for s in samples if s[0] in train_games]
    test = [s for s in samples if s[0] not in train_games]
    print(f"{len(train)} train positions, {len(test)} test positions")

    train_loader = DataLoader(ChessMoveDataset(train), batch_size=64, shuffle=True)
    test_loader = DataLoader(ChessMoveDataset(test), batch_size=256)

    # Sanity check: can the ResNet memorize 64 positions?
    tiny = DataLoader(ChessMoveDataset(train[:64]), batch_size=64)
    m = ChessResNet().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for step in range(200):
        train_one_epoch(m, tiny, opt, device)
    print("overfit-64 accuracy (should be ~1.0):", evaluate(m, tiny, device))

    for name, model in [("SimpleCNN", SimpleCNN()), ("ChessResNet", ChessResNet())]:
        model = model.to(device)
        print(f"\n{name}: {count_params(model):,} parameters")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(1, 11):
            loss = train_one_epoch(model, train_loader, optimizer, device)
            if epoch % 2 == 0:
                t1, t3 = evaluate(model, test_loader, device)
                print(f"epoch {epoch:2d}  loss {loss:.3f}  test top-1 {t1:.1%}  top-3 {t3:.1%}")
        print("prediction for the start position:", predict(model, chess.STARTING_FEN, device=device))


if __name__ == "__main__":
    main(sys.argv[1])
```

Run it:

```bash
python src/chesspred/tutorial.py data/chess_com_games_2026-11-04.pgn
```

### What you should see

This is from a test run on **your 50 games** (CPU, ~2 minutes):

```
Round-trip test passed on 3181 moves
2554 train positions, 627 test positions
overfit-64 accuracy (should be ~1.0): (0.96875, 1.0)

SimpleCNN: 16,865,600 parameters
epoch  2  loss 2.655  test top-1 17.1%  top-3 34.0%
epoch  6  loss 2.156  test top-1 14.0%  top-3 33.0%
epoch 10  loss 1.311  test top-1 13.9%  top-3 32.9%
prediction for the start position: [('Nf3', 0.278), ('e4', 0.228), ('d4', 0.205)]

ChessResNet: 8,849,216 parameters
epoch  2  loss 2.570  test top-1 15.8%  top-3 35.1%
epoch  6  loss 1.139  test top-1 16.9%  top-3 35.7%
epoch 10  loss 0.312  test top-1 18.0%  top-3 35.9%
prediction for the start position: [('Nf3', 0.525), ('e4', 0.343), ('d4', 0.088)]
```

| Model | Test top-1 | Test top-3 |
|---|---|---|
| Random legal move | 6.1% | 15.2% |
| SimpleCNN (epoch 10) | 13.9% | 32.9% |
| ChessResNet (epoch 10) | 18.0% | 35.9% |

Your exact numbers will differ a little (random initialization, GPU vs CPU).

### How to read this

- **Both models beat random** (6.1%), so the pipeline works. 🎉
- **The start-position predictions are your openings.** Nf3, e4, and d4 are exactly what you play as White. The model picked that up from just 40 games.
- **The training loss keeps falling, but test accuracy stalls.** That's **overfitting**: with only ~2,500 positions, the models memorize the training games instead of learning chess. **This is expected, and it's why the roadmap trains on millions of Lichess positions first** and only then fine-tunes on your games.
- **The ResNet does better with half the parameters**, which is why it's the core model.

---

## Debugging checklist

Two tests save hours. They're both in `main()` already:

1. **Round-trip test**: every real move must be legal in the mask and decode back to itself. If this fails, your encoding has a bug (usually the Black mirroring).
2. **Overfit-64 test**: a model that **can't memorize 64 positions** has a bug in the model, the loss, or the labels. It should reach close to 100%.

| Symptom | Likely cause |
|---|---|
| Loss is `nan` | The real move was masked out (encoding bug), so the round-trip test should catch it. Or the learning rate is too high. |
| Accuracy near 3% after training | Labels don't line up with positions (for example, the board was pushed *before* the sample was saved). |
| Good as White, terrible as Black | The move wasn't mirrored the same way as the board. |
| ResNet worse at test time than during training | You forgot `model.eval()` (BatchNorm). |
| Training is slow even on the GPU | The data loader is the bottleneck. See the next section. |

---

## Moving to the RTX 3060 / Colab

The code above already picks `cuda` automatically. For real training (roadmap weeks 2–3), make four changes:

1. **Bigger batches:** `batch_size=1024` (12 GB of VRAM handles this easily at 6 × 64).
2. **Parallel data loading:** `DataLoader(..., num_workers=4, pin_memory=True)`.
3. **Mixed precision**, for faster training and less memory:
   ```python
   with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
       logits = masked(model(x), mask)
       loss = F.cross_entropy(logits, y)
   ```
4. **Pre-encode the data once.** Convert all positions to arrays (`X: uint8 [N,18,8,8]`, `y: int16 [N]`) and save them with `np.save`, instead of parsing FENs inside `__getitem__`. Build the legal masks on the fly, or store the legal move indices.

Save a checkpoint every epoch so Colab disconnects don't lose work:

```python
torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
           "checkpoint.pt")
```

---

## Exercises (each one is a step toward the roadmap)

1. **Only your moves:** evaluate with `load_positions(path, player="whopizza1")` for the test set. That's the headline metric.
2. **Excluding openings:** skip positions where `board.fullmove_number <= 8` and compare. This is the "openings vs. the rest" section of your final report.
3. **Make it bigger:** try `ChessResNet(blocks=10, channels=128)`. On your 50 games it will overfit more, which shows why more data comes first.
4. **Real data:** swap the PGN for a filtered Lichess month (roadmap week 1) and watch the test accuracy climb instead of stalling.
5. **Fine-tune:** train on Lichess, then continue training on your games with a lower learning rate (`lr=1e-4`). That's roadmap week 3.
6. **Elo conditioning (model 1.4):** add `nn.Embedding` layers for player and opponent rating buckets and add them to the features after the stem.
