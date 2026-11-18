"""Round-trip every move in your PGN through encoding (castling, en passant, promotion, Black flips).

Run from the repo root:  pytest -q
"""

from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pytest
import torch

from chesspred.encoding import (
    NUM_PLANES,
    bitboards_to_numpy,
    bitboards_to_planes,
    board_to_bitboards,
    board_to_tensor,
    encode_chunk,
    index_to_move,
    legal_mask_from_padded,
    legal_move_mask,
    move_to_index,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PGN = REPO_ROOT / "data" / "raw" / "chess_com_games_2026-11-04.pgn"


def reference_board_to_tensor(board: chess.Board) -> torch.Tensor:
    """The tutorial's straightforward version (board.mirror + loops), used as ground truth."""
    white_to_move = board.turn == chess.WHITE
    if not white_to_move:
        board = board.mirror()
    x = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        row, col = divmod(square, 8)
        x[plane, row, col] = 1.0
    x[12] = 1.0 if white_to_move else 0.0
    x[13] = board.has_kingside_castling_rights(chess.WHITE)
    x[14] = board.has_queenside_castling_rights(chess.WHITE)
    x[15] = board.has_kingside_castling_rights(chess.BLACK)
    x[16] = board.has_queenside_castling_rights(chess.BLACK)
    if board.has_legal_en_passant():
        row, col = divmod(board.ep_square, 8)
        x[17, row, col] = 1.0
    return torch.from_numpy(x)


def pgn_positions() -> list[tuple[chess.Board, chess.Move]]:
    if not PGN.exists():
        pytest.skip(f"{PGN.name} not found (data/ is gitignored)")
    out = []
    with open(PGN) as f:
        while (game := chess.pgn.read_game(f)) is not None:
            board = game.board()
            for move in game.mainline_moves():
                out.append((board.copy(stack=False), move))
                board.push(move)
    return out


# Hand-picked positions for the tricky cases.
SPECIAL_FENS = {
    "start": chess.STARTING_FEN,
    "black to move after 1.e4": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
    "white en passant": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    "black en passant": "rnbqkbnr/pppp1ppp/8/8/3Pp3/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 3",
    "castling both ways": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
    "black castling, partial rights": "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b Kq - 0 1",
    "white promotion": "8/P6k/8/8/8/8/7K/8 w - - 0 1",
    "black promotion with capture": "8/7k/8/8/8/8/1p5K/R7 b - - 0 1",
}


@pytest.mark.parametrize("name", SPECIAL_FENS)
def test_special_positions_match_reference(name):
    board = chess.Board(SPECIAL_FENS[name])
    assert torch.equal(board_to_tensor(board), reference_board_to_tensor(board)), name
    for move in board.legal_moves:
        idx = move_to_index(move, board.turn)
        assert legal_move_mask(board)[idx]
        if move.promotion in (None, chess.QUEEN):
            assert index_to_move(idx, board) == move


def test_black_view_is_mirror_of_white_view():
    # After 1.e4 e5 2.Nf3 Nc6, both sides have made mirror-image moves, so the piece
    # planes must look identical from either side's point of view.
    white = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    black = white.mirror()
    assert torch.equal(board_to_tensor(white)[:12], board_to_tensor(black)[:12])


def test_every_pgn_position_matches_reference():
    positions = pgn_positions()
    bbs = torch.from_numpy(bitboards_to_numpy([board_to_bitboards(b) for b, _ in positions]))
    planes = bitboards_to_planes(bbs)
    for i, (board, _) in enumerate(positions):
        assert torch.equal(planes[i], reference_board_to_tensor(board)), board.fen()


def test_every_pgn_move_round_trips():
    positions = pgn_positions()
    for board, move in positions:
        idx = move_to_index(move, board.turn)
        assert legal_move_mask(board)[idx], (board.fen(), move.uci())
        if move.promotion in (None, chess.QUEEN):
            assert index_to_move(idx, board) == move, (board.fen(), move.uci())
    print(f"round-trip passed on {len(positions)} moves")


def test_preencoded_chunk_matches_single_position_functions():
    positions = pgn_positions()[:500]
    fens = [b.fen() for b, _ in positions]
    ucis = [m.uci() for _, m in positions]
    boards, labels, legal = encode_chunk((fens, ucis))
    assert (labels >= 0).all()
    masks = legal_mask_from_padded(torch.from_numpy(legal))
    planes = bitboards_to_planes(torch.from_numpy(boards))
    for i, (board, move) in enumerate(positions):
        assert torch.equal(masks[i], legal_move_mask(board))
        assert torch.equal(planes[i], board_to_tensor(board))
        assert labels[i] == move_to_index(move, board.turn)
