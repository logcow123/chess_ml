"""Board -> 8x8x18 tensor, move <-> 4096 index, legal-move mask.

Same design as docs/cnn_resnet_tutorial.md (roadmap decisions #4 and #5):

    planes 0-5    our   P N B R Q K       (1 where that piece is)
    planes 6-11   their P N B R Q K
    plane  12     all 1s if we are White
    planes 13-16  castling: ours K-side, ours Q-side, theirs K-side, theirs Q-side
    plane  17     en passant target square (only when an en passant capture is legal)

Always from the side-to-move's point of view: when Black is to move the board is flipped
vertically and the colours swapped, so "us" is always White moving up the board.

Two ways to build the planes, which always give the same answer:

    board_to_tensor(board)        one position -> float tensor (18, 8, 8). Used by the app.
    board_to_bitboards(board)     one position -> 18 Python ints, one 64-bit bitboard per plane
    bitboards_to_planes(bb)       batch (N, 18) int64 -> float tensor (N, 18, 8, 8), on any device

Training stores the 18 bitboards (144 bytes per position) and unpacks them on the GPU. That
keeps 7.5M Lichess positions at about 1 GB instead of ~35 GB of float planes, small enough
to live on the RTX 3060 for the whole run, so the data loader is never the bottleneck.

Pre-encode a parsed parquet once (from the repo root, after `pip install -e .`):

    python -m chesspred.encoding data/processed/lichess_2025-09_blitz-rapid_900-1400_rapid.parquet
    python -m chesspred.encoding data/processed/chesscom_whopizza1_2026-11-06_rapid_whopizza1-moves.parquet

Each writes a .npz with the same name next to the parquet. train.py reads those files.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import zlib
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import chess
import numpy as np
import torch

NUM_PLANES = 18
NUM_MOVES = 64 * 64  # from-square x to-square
PAD_MOVE = NUM_MOVES  # filler in padded legal-move lists (a dummy 4,097th column)
ALL_SQUARES = chess.BB_ALL  # a bitboard with every square set: a plane of 1s

# Split codes stored in the .npz (train.py only ever trains on 0 and validates on 1).
SPLIT_CODES = {"train": 0, "val": 1, "test": 2, "unassigned": 3}


# --------------------------------------------------------------------------------------
# Board -> planes
# --------------------------------------------------------------------------------------
def _same(bb: int) -> int:
    return bb


def board_to_bitboards(board: chess.Board) -> list[int]:
    """The 18 planes as 64-bit ints. Bit s is square s (a1 = 0, b1 = 1, ..., h8 = 63)."""
    us = board.turn
    them = not us
    # Flipping the ranks + swapping who is "us" is exactly what board.mirror() does,
    # but done on the bitboards directly so we don't have to copy the board.
    flip = chess.flip_vertical if us == chess.BLACK else _same

    planes = [flip(board.pieces_mask(pt, us)) for pt in chess.PIECE_TYPES]     # 0-5
    planes += [flip(board.pieces_mask(pt, them)) for pt in chess.PIECE_TYPES]  # 6-11
    planes.append(ALL_SQUARES if us == chess.WHITE else 0)                     # 12
    for color in (us, them):                                                   # 13-16
        planes.append(ALL_SQUARES if board.has_kingside_castling_rights(color) else 0)
        planes.append(ALL_SQUARES if board.has_queenside_castling_rights(color) else 0)
    ep = 0                                                                     # 17
    if board.has_legal_en_passant():
        sq = board.ep_square if us == chess.WHITE else chess.square_mirror(board.ep_square)
        ep = chess.BB_SQUARES[sq]
    planes.append(ep)
    return planes


def bitboards_to_numpy(bitboards: list[list[int]] | list[int]) -> np.ndarray:
    """Python ints -> int64 array. (uint64 first, because h8 is bit 63, the sign bit.)"""
    return np.asarray(bitboards, dtype=np.uint64).view(np.int64)


_SHIFTS: dict[torch.device, torch.Tensor] = {}


def bitboards_to_planes(bitboards: torch.Tensor) -> torch.Tensor:
    """(N, 18) int64 bitboards -> (N, 18, 8, 8) float32 planes, on the same device.

    (x >> s) & 1 pulls out bit s. Square s lands at row s // 8 (rank), column s % 8 (file),
    the same layout as the tutorial's `row, col = divmod(square, 8)`.
    """
    device = bitboards.device
    if device not in _SHIFTS:
        _SHIFTS[device] = torch.arange(64, dtype=torch.int64, device=device)
    bits = (bitboards.unsqueeze(-1) >> _SHIFTS[device]) & 1  # (N, 18, 64)
    return bits.view(-1, NUM_PLANES, 8, 8).to(torch.float32)


def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """Encode one board as an (18, 8, 8) float tensor, from the side-to-move's view."""
    bb = torch.from_numpy(bitboards_to_numpy([board_to_bitboards(board)]))
    return bitboards_to_planes(bb)[0]


# --------------------------------------------------------------------------------------
# Moves <-> class indices
# --------------------------------------------------------------------------------------
def move_to_index(move: chess.Move, turn: chess.Color) -> int:
    """Map a move to one of 4096 classes (from * 64 + to), in the mover's point of view."""
    from_sq, to_sq = move.from_square, move.to_square
    if turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    return from_sq * 64 + to_sq


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of move_to_index for the given board (promotions become queen promotions)."""
    from_sq, to_sq = divmod(int(index), 64)
    if board.turn == chess.BLACK:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)
    move = chess.Move(from_sq, to_sq)
    piece = board.piece_at(from_sq)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
        move.promotion = chess.QUEEN
    return move


def legal_move_indices(board: chess.Board) -> list[int]:
    """Class indices of every legal move (underpromotions share the queen's index)."""
    return list(dict.fromkeys(move_to_index(m, board.turn) for m in board.legal_moves))


def legal_move_mask(board: chess.Board) -> torch.Tensor:
    """(4096,) bool: True for legal moves."""
    mask = torch.zeros(NUM_MOVES, dtype=torch.bool)
    mask[legal_move_indices(board)] = True
    return mask


def legal_mask_from_padded(legal: torch.Tensor) -> torch.Tensor:
    """(N, K) padded legal-move indices (filler = PAD_MOVE) -> (N, 4096) bool mask.

    The filler points at an extra 4,097th column that is thrown away afterwards.
    """
    mask = torch.zeros(legal.shape[0], NUM_MOVES + 1, dtype=torch.bool, device=legal.device)
    mask.scatter_(1, legal.long(), True)
    return mask[:, :NUM_MOVES]


# --------------------------------------------------------------------------------------
# Pre-encoding: parsed parquet -> compact .npz for train.py
# --------------------------------------------------------------------------------------
def encode_chunk(rows: tuple[list[str], list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode a chunk of (fen, move_uci) rows. Runs inside a worker process.

    Returns bitboards (n, 18) int64, labels (n,) int16 (-1 = bad row, dropped later)
    and legal-move indices (n, K) int16 padded with PAD_MOVE.
    """
    fens, ucis = rows
    n = len(fens)
    boards = np.zeros((n, NUM_PLANES), dtype=np.uint64)
    labels = np.full(n, -1, dtype=np.int16)
    legal_lists = []
    for i, (fen, uci) in enumerate(zip(fens, ucis)):
        board = chess.Board(fen)
        boards[i] = board_to_bitboards(board)
        legal = legal_move_indices(board)
        label = move_to_index(chess.Move.from_uci(uci), board.turn)
        if label in legal:  # always true for clean data; a False here means a parsing bug
            labels[i] = label
        legal_lists.append(legal)

    width = max(len(m) for m in legal_lists)
    legal_arr = np.full((n, width), PAD_MOVE, dtype=np.int16)
    for i, moves in enumerate(legal_lists):
        legal_arr[i, : len(moves)] = moves
    return boards.view(np.int64), labels, legal_arr


def assign_split(game_ids: list[str], method: str = "auto",
                 val_pct: float = 2.0) -> tuple[np.ndarray, str]:
    """Split code for each (unique) game. Always by GAME, never by position (roadmap §8).

    frozen  your games: the frozen date-based split from data/splits/ (train/val/test).
    hash    Lichess: a stable ~val_pct% of games (by a hash of the game id) become val.
    auto    frozen if most of these games are in the frozen split file, else hash.
            This stops your test games from ever leaking into training by accident.
    """
    from chesspred.data.split import DEFAULT_SPLIT, load_split

    frozen = load_split(DEFAULT_SPLIT) if DEFAULT_SPLIT.exists() else {}
    if method == "auto":
        in_frozen = sum(g in frozen for g in game_ids)
        method = "frozen" if game_ids and in_frozen > 0.5 * len(game_ids) else "hash"

    if method == "frozen":
        codes = [SPLIT_CODES[frozen.get(g, "unassigned")] for g in game_ids]
    else:
        codes = [SPLIT_CODES["val"] if zlib.crc32(g.encode()) % 10_000 < val_pct * 100
                 else SPLIT_CODES["train"] for g in game_ids]
    return np.array(codes, dtype=np.uint8), method


def preencode(parquet_path: Path, out_path: Path, split: str = "auto", val_pct: float = 2.0,
              workers: int | None = None, chunk_size: int = 20_000) -> dict:
    """Encode every position in a parsed parquet and save the arrays as one .npz.

    Streams the FEN/move columns in chunks, so millions of positions never sit in memory
    as Python strings. Peak RAM is roughly 3x the size of the finished .npz.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n = pf.metadata.num_rows

    # Per-row metadata, read without making millions of Python objects.
    table = pf.read(columns=["game_id", "player_elo", "opponent_elo", "move_number"])
    games = table["game_id"].combine_chunks().dictionary_encode()  # factorize in Arrow
    game_idx = games.indices.to_numpy(zero_copy_only=False).astype(np.int32)
    game_ids = games.dictionary.to_pylist()
    split_per_game, split_method = assign_split(game_ids, split, val_pct)
    split_codes = split_per_game[game_idx]
    elo = np.stack([pc.fill_null(table[c], -1).to_numpy() for c in ("player_elo", "opponent_elo")],
                   axis=1).astype(np.int16)  # -1 = unknown rating
    move_number = table["move_number"].to_numpy().astype(np.int16)
    del table, games

    def fen_chunks():
        for batch in pf.iter_batches(batch_size=chunk_size, columns=["fen", "move_uci"]):
            yield batch.column(0).to_pylist(), batch.column(1).to_pylist()

    boards = np.zeros((n, NUM_PLANES), dtype=np.int64)
    labels = np.zeros(n, dtype=np.int16)
    legal_chunks = []
    row = 0
    t0 = time.time()

    def store(result) -> None:
        nonlocal row
        b, lab, leg = result
        boards[row:row + len(b)] = b
        labels[row:row + len(b)] = lab
        legal_chunks.append(leg)
        row += len(b)
        if len(legal_chunks) % 25 == 0 or row == n:
            print(f"  encoded {row:,}/{n:,} positions ({time.time() - t0:,.0f}s)", flush=True)

    workers = workers or os.cpu_count() or 1
    if workers == 1:
        for chunk in fen_chunks():
            store(encode_chunk(chunk))
    else:
        # Feed the pool a few chunks per worker at a time (Pool.imap alone would read the
        # whole file into its task queue up front).
        chunks = fen_chunks()
        with Pool(workers) as pool:
            while group := list(islice(chunks, workers * 4)):
                for result in pool.imap(encode_chunk, group):
                    store(result)

    # Pad every chunk's legal-move list to the widest one seen.
    width = max(c.shape[1] for c in legal_chunks)
    legal = np.full((n, width), PAD_MOVE, dtype=np.int16)
    row = 0
    legal_chunks.reverse()
    while legal_chunks:  # pop as we go, so the chunks are freed while copying
        c = legal_chunks.pop()
        legal[row:row + len(c), : c.shape[1]] = c
        row += len(c)

    arrays = {
        "boards": boards,                 # (N, 18) int64 bitboards
        "labels": labels,                 # (N,) int16 move index played
        "legal": legal,                   # (N, K) int16 legal moves, PAD_MOVE filler
        "elo": elo,                       # (N, 2) int16 mover, opponent (-1 = unknown)
        "move_number": move_number,       # (N,) int16, for "excluding openings" / by-phase eval
        "split": split_codes,             # (N,) uint8, see SPLIT_CODES
        "game_idx": game_idx,             # (N,) int32 row -> index into game_ids
    }
    keep = labels >= 0
    dropped = int((~keep).sum())
    if dropped:  # a played move that isn't legal means a parsing bug; drop those rows
        arrays = {k: v[keep] for k, v in arrays.items()}

    sc = arrays["split"]
    meta = {
        "source": parquet_path.name,
        "positions": int(len(sc)),
        "dropped_bad_rows": dropped,
        "split_method": split_method,
        "split_counts": {name: int((sc == code).sum()) for name, code in SPLIT_CODES.items()},
        "legal_width": int(width),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    with open(tmp, "wb") as f:  # write under a temp name, so a crash never leaves a half file
        np.savez(f, **arrays, game_ids=np.asarray(game_ids, dtype=str),
                 meta=np.array(json.dumps(meta)))
    tmp.replace(out_path)
    meta["seconds"] = round(time.time() - t0, 1)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parquet", type=Path, help="parsed positions from parse_pgn.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .npz (default: same name as the parquet, next to it)")
    parser.add_argument("--split", choices=["auto", "frozen", "hash"], default="auto",
                        help="auto (default): frozen split for your games, hash split otherwise")
    parser.add_argument("--val-pct", type=float, default=2.0,
                        help="percent of games used for validation with the hash split (default 2)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: all CPU cores; 1 = no multiprocessing)")
    args = parser.parse_args()

    out = args.out or args.parquet.with_suffix(".npz")
    print(f"Pre-encoding {args.parquet}\n  -> {out}")
    meta = preencode(args.parquet, out, args.split, args.val_pct, args.workers)
    print(f"\nDone in {meta['seconds']:,}s: {meta['positions']:,} positions "
          f"({meta['dropped_bad_rows']} bad rows dropped)")
    print(f"  split ({meta['split_method']}): {meta['split_counts']}")
    print(f"  max legal moves in one position: {meta['legal_width']}")
    print(f"  file: {out}  ({out.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
