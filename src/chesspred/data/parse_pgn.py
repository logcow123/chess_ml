"""Parse chess games into one row per move, keeping only RAPID games (by default).

Every output row is one position plus the move that was actually played from it:

    game_id, source, date, speed, time_control, result,
    ply, move_number, side, player, player_elo, opponent_elo, fen, move_uci

The roadmap's core columns are fen, move_uci, player_elo, opponent_elo, time_control,
move_number and game_id. The extras are there for the date-based test split (`date`),
EDA (`result`, `speed`) and for filtering to one person's moves (`player`, `side`).

Input formats (picked automatically from the file extension):
  * .jsonl    chess.com API dump made by download_chesscom.py. Uses chess.com's own
              `time_class` label, so "rapid" means exactly what chess.com says.
  * .pgn      any plain PGN export (chess.com or Lichess).
  * .parquet  Lichess dump made by download_lichess.py (uses its `speed` column).

Usage (from the repo root, after `pip install -e .`):
    # Your chess.com games, keeping only the moves YOU played (for fine-tuning)
    python -m chesspred.data.parse_pgn data/raw/chesscom_whopizza1_2026-11-06.jsonl --player whopizza1

    # Lichess rapid games, both sides (for pre-training)
    python -m chesspred.data.parse_pgn data/raw/lichess_2025-09_blitz-rapid_900-1400.parquet

    # Quick test on the first 500 games, or keep blitz too
    python -m chesspred.data.parse_pgn data/raw/lichess_2025-09_blitz-rapid_900-1400.parquet --max-games 500
    python -m chesspred.data.parse_pgn data/raw/chess_com_games_2026-11-04.pgn --speeds rapid blitz

Output goes to data/processed/<input name>_<speeds>[_<player>-moves].parquet unless --out
is given. Load it back with:  df = load_positions("data/processed/....parquet")

Requires: python-chess, pyarrow, pandas (all in requirements.txt).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator

import chess
import chess.pgn
import pyarrow as pa
import pyarrow.parquet as pq


# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
def find_repo_root() -> Path:
    """Walk up from this file to the folder holding pyproject.toml (the repo root).

    Same trick as download_lichess.py, so default paths work from any working directory.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


REPO_ROOT = find_repo_root()
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"


# --------------------------------------------------------------------------------------
# Output schema
# --------------------------------------------------------------------------------------
# Order matters: the worker returns plain tuples in exactly this order (tuples pickle
# much faster than dicts when sending millions of rows back from worker processes).
SCHEMA = pa.schema([
    ("game_id", pa.string()),       # chess.com/Lichess game URL, or a hash if there's no URL
    ("source", pa.string()),        # "chesscom" or "lichess"
    ("date", pa.string()),          # UTC "YYYY-MM-DD HH:MM:SS" (or just the date); sorts correctly as text
    ("speed", pa.string()),         # "rapid", "blitz", ...
    ("time_control", pa.string()),  # raw tag, e.g. "600" or "600+5"
    ("result", pa.string()),        # "1-0", "0-1", "1/2-1/2"
    ("ply", pa.int16()),            # 0-based half-move index: 0 = White's first move
    ("move_number", pa.int16()),    # the "12" in "12. Nf3" / "12... Nf6"
    ("side", pa.string()),          # "w" or "b": who is moving in this position
    ("player", pa.string()),        # username of the player making this move
    ("player_elo", pa.int16()),     # rating of the player making this move (null if unknown)
    ("opponent_elo", pa.int16()),   # rating of the other player (null if unknown)
    ("fen", pa.string()),           # position BEFORE the move (full FEN)
    ("move_uci", pa.string()),      # the move played, e.g. "e2e4", "e7e8q", "e1g1" (castling)
])
COLUMNS = SCHEMA.names

FLUSH_EVERY = 250_000  # rows kept in memory before appending to the parquet file


# --------------------------------------------------------------------------------------
# Speed classification ("is this game rapid?")
# --------------------------------------------------------------------------------------
# Both sites estimate how long a game lasts as  base + 40 * increment  (seconds, assuming
# a 40-move game) and bucket it. Checked against the real labels in data/raw/:
#   chess.com: 120+1 -> 160s bullet, 300 -> blitz, 600 / 900+10 / 1800 -> rapid
#   Lichess:   300+8 -> 620s rapid,  600+0 -> rapid, 1500s and up -> classical
# Buckets are (upper bound in seconds, exclusive, speed name), checked in order.
CHESSCOM_BUCKETS = [(180, "bullet"), (600, "blitz"), (float("inf"), "rapid")]
LICHESS_BUCKETS = [(30, "ultrabullet"), (180, "bullet"), (480, "blitz"),
                   (1500, "rapid"), (float("inf"), "classical")]

# Lichess writes the speed into the Event tag ("Rated Rapid game"); chess.com doesn't.
EVENT_SPEED_RE = re.compile(
    r"\b(ultrabullet|bullet|blitz|rapid|classical|correspondence)\b", re.IGNORECASE)


def estimated_seconds(time_control: str | None) -> float | None:
    """'600+5' -> 600 + 40*5 = 800 seconds.

    Returns None when there's no clock to estimate: daily/correspondence games
    ('1/259200' = one move per 3 days) and missing tags ('-', '?', '').
    """
    tc = (time_control or "").strip()
    if not tc or tc in {"-", "?"} or "/" in tc:
        return None
    base, _, inc = tc.partition("+")
    try:
        return float(base) + 40 * float(inc or 0)
    except ValueError:
        return None


def classify_speed(time_control: str | None, site: str = "chesscom", event: str = "") -> str:
    """Return 'bullet' / 'blitz' / 'rapid' / ... for a game.

    Priority: an explicit label in the Event tag (Lichess), then the time-control
    estimate using the site's own buckets. `site` is "chesscom" or "lichess".
    """
    match = EVENT_SPEED_RE.search(event or "")
    if match:
        return match.group(1).lower()
    if time_control and "/" in time_control:
        return "daily"
    est = estimated_seconds(time_control)
    if est is None:
        return "unknown"
    buckets = LICHESS_BUCKETS if site == "lichess" else CHESSCOM_BUCKETS
    for upper, name in buckets:
        if est < upper:
            return name
    return "unknown"  # unreachable: the last bucket is open-ended


# --------------------------------------------------------------------------------------
# Small helpers for messy header values
# --------------------------------------------------------------------------------------
def to_int(value) -> int | None:
    """Ratings arrive as ints, '1234' strings, or '?' — turn them into int or None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def site_of(text: str) -> str:
    """Guess which site a game came from by looking at its Site/Link header text."""
    return "lichess" if "lichess" in (text or "").lower() else "chesscom"


def pgn_date(headers: dict) -> str:
    """Best UTC timestamp available in a PGN's headers, as 'YYYY-MM-DD[ HH:MM:SS]'.

    chess.com API PGNs have UTCDate/UTCTime. Older chess.com exports only have Date
    plus EndTime ("21:38:52 GMT+0000"). Lichess PGNs have UTCDate/UTCTime.
    """
    day = headers.get("UTCDate") or headers.get("Date") or ""
    clock = headers.get("UTCTime") or (headers.get("EndTime") or "").split(" ")[0]
    day = day.replace(".", "-")
    if "?" in day or not day:
        return ""
    return f"{day} {clock}".strip()


# --------------------------------------------------------------------------------------
# Readers: each yields one dict per game with the same keys, already speed-filtered.
# The expensive part (replaying the moves) happens later in the worker processes, so the
# readers only look at cheap metadata and skip non-rapid games before any move parsing.
# --------------------------------------------------------------------------------------
# A "game info" dict has: game_id, source, date, speed, time_control, result,
# white, black, white_elo, black_elo, pgn (full PGN text or bare movetext).

def iter_chesscom_jsonl(path: Path, speeds: set[str], stats: Counter) -> Iterator[dict]:
    """Games from download_chesscom.py output (one monthly archive per line)."""
    with open(path) as f:
        for line in f:
            for g in json.loads(line).get("games", []):
                stats["games_seen"] += 1
                # Variants (chess960, crazyhouse, ...) can't share a model with normal chess.
                if g.get("rules", "chess") != "chess":
                    stats["skipped_variant"] += 1
                    continue
                speed = g.get("time_class") or classify_speed(g.get("time_control"))
                stats[f"speed_{speed}"] += 1
                if speed not in speeds:
                    continue
                end = g.get("end_time")
                date = (datetime.fromtimestamp(end, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        if end else "")
                white, black = g.get("white", {}), g.get("black", {})
                yield {
                    "game_id": g.get("url") or g.get("uuid") or "",
                    "source": "chesscom",
                    "date": date,
                    "speed": speed,
                    "time_control": g.get("time_control", ""),
                    "result": None,  # read from the PGN's Result tag in the worker
                    "white": white.get("username", ""),
                    "black": black.get("username", ""),
                    "white_elo": to_int(white.get("rating")),
                    "black_elo": to_int(black.get("rating")),
                    "pgn": g.get("pgn", ""),
                }


# Games in a PGN file are separated by a blank line followed by the next game's "[Event".
PGN_SPLIT_RE = re.compile(r"\n\s*\n(?=\[Event )")
HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$', re.MULTILINE)


def iter_pgn_file(path: Path, speeds: set[str], stats: Counter) -> Iterator[dict]:
    """Games from a plain .pgn file (chess.com or Lichess export)."""
    text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    for chunk in PGN_SPLIT_RE.split(text.strip()):
        if not chunk.strip():
            continue
        stats["games_seen"] += 1
        headers = dict(HEADER_RE.findall(chunk))  # fast regex peek; full parse happens later
        if headers.get("Variant", "Standard").lower() not in {"standard", "chess"}:
            stats["skipped_variant"] += 1
            continue
        link = headers.get("Link") or headers.get("Site", "")
        site = site_of(link)
        speed = classify_speed(headers.get("TimeControl"), site, headers.get("Event", ""))
        stats[f"speed_{speed}"] += 1
        if speed not in speeds:
            continue
        yield {
            # Only use the link as an id if it points at a specific game.
            "game_id": link if "/game" in link or "lichess.org/" in link else "",
            "source": site,
            "date": pgn_date(headers),
            "speed": speed,
            "time_control": headers.get("TimeControl", ""),
            "result": headers.get("Result"),
            "white": headers.get("White", ""),
            "black": headers.get("Black", ""),
            "white_elo": to_int(headers.get("WhiteElo")),
            "black_elo": to_int(headers.get("BlackElo")),
            "pgn": chunk,
        }


LICHESS_COLUMNS = ["Site", "Event", "White", "Black", "Result", "WhiteElo", "BlackElo",
                   "UTCDate", "TimeControl", "movetext", "speed"]


def iter_lichess_parquet(path: Path, speeds: set[str], stats: Counter) -> Iterator[dict]:
    """Games from download_lichess.py output (one row per game, moves in `movetext`)."""
    pf = pq.ParquetFile(path)
    columns = [c for c in LICHESS_COLUMNS if c in pf.schema_arrow.names]
    # Read in batches so a multi-GB file never has to fit in memory at once.
    for batch in pf.iter_batches(batch_size=20_000, columns=columns):
        for r in batch.to_pylist():
            stats["games_seen"] += 1
            # Prefer the `speed` column download_lichess.py added; fall back to Event/TC.
            speed = (r.get("speed") or "").lower() or classify_speed(
                r.get("TimeControl"), "lichess", r.get("Event", ""))
            stats[f"speed_{speed}"] += 1
            if speed not in speeds:
                continue
            day = r.get("UTCDate")
            yield {
                "game_id": r.get("Site", ""),
                "source": "lichess",
                "date": str(day) if day else "",
                "speed": speed,
                "time_control": r.get("TimeControl", ""),
                "result": r.get("Result"),
                "white": r.get("White", ""),
                "black": r.get("Black", ""),
                "white_elo": to_int(r.get("WhiteElo")),
                "black_elo": to_int(r.get("BlackElo")),
                "pgn": r.get("movetext", ""),  # bare movetext parses fine as a PGN
            }


def iter_games(path: Path, speeds: set[str], stats: Counter) -> Iterator[dict]:
    """Pick the right reader for a file based on its extension."""
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return iter_chesscom_jsonl(path, speeds, stats)
    if suffix == ".pgn":
        return iter_pgn_file(path, speeds, stats)
    if suffix == ".parquet":
        return iter_lichess_parquet(path, speeds, stats)
    raise SystemExit(f"Don't know how to read {path} (expected .jsonl, .pgn or .parquet)")


# --------------------------------------------------------------------------------------
# Worker: replay one game move by move and emit a row per position
# --------------------------------------------------------------------------------------
def game_to_rows(info: dict, player: str | None = None,
                 min_moves: int = 0) -> tuple[str, str, list[tuple]]:
    """Turn one game into rows. Runs inside a worker process.

    Returns (status, dedup_key, rows). status is "ok" or a skip reason that the main
    process tallies for the summary.
    """
    game = chess.pgn.read_game(io.StringIO(info["pgn"]))
    # python-chess doesn't raise on bad PGN; it records problems in game.errors.
    if game is None or game.errors:
        return "skipped_parse_error", "", []

    board = game.board()
    # A FEN/SetUp header means a custom start (odds games, chess960 exports): skip.
    if board.fen() != chess.STARTING_FEN:
        return "skipped_nonstandard_start", "", []

    moves = list(game.mainline_moves())
    if (len(moves) + 1) // 2 < min_moves:  # plies -> full moves, rounding up
        return "skipped_too_short", "", []

    white, black = info["white"], info["black"]
    result = info["result"] or game.headers.get("Result", "*")
    want = player.lower() if player else None

    # The same game can show up in two files (e.g. the PGN export AND the API dump),
    # with different ids. Players + exact move list identifies it regardless of source.
    key_text = f"{white.lower()}|{black.lower()}|{' '.join(m.uci() for m in moves)}"
    dedup_key = hashlib.sha1(key_text.encode()).hexdigest()

    # Fall back to that hash as the id when the file had no game URL.
    game_id = info["game_id"] or f"hash:{dedup_key[:16]}"

    rows = []
    for ply, move in enumerate(moves):
        white_to_move = board.turn == chess.WHITE
        mover = white if white_to_move else black
        # --player: keep only positions where that person is the one choosing the move.
        # We still push every move so the board stays correct.
        if want is None or mover.lower() == want:
            rows.append((
                game_id,
                info["source"],
                info["date"],
                info["speed"],
                info["time_control"],
                result,
                ply,
                board.fullmove_number,
                "w" if white_to_move else "b",
                mover,
                info["white_elo"] if white_to_move else info["black_elo"],
                info["black_elo"] if white_to_move else info["white_elo"],
                board.fen(),   # position BEFORE the move...
                move.uci(),    # ...and the move that was played from it
            ))
        board.push(move)

    if want is not None and not rows:
        return "skipped_player_not_in_game", dedup_key, []
    return "ok", dedup_key, rows


# --------------------------------------------------------------------------------------
# Writer: stream rows into one parquet file in chunks
# --------------------------------------------------------------------------------------
class ParquetSink:
    """Collects row tuples and appends them to a parquet file every FLUSH_EVERY rows.

    Writing in chunks keeps memory flat even for millions of positions. The file is
    written to a .part name first and only renamed when finished, so a crash never
    leaves a half-written file that looks complete.
    """

    def __init__(self, out_path: Path):
        self.out_path = out_path
        self.tmp_path = out_path.with_suffix(".parquet.part")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(self.tmp_path, SCHEMA, compression="zstd")
        self.buffer: list[tuple] = []
        self.total = 0

    def add(self, rows: list[tuple]) -> None:
        self.buffer.extend(rows)
        if len(self.buffer) >= FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        # Transpose row tuples into one column list per field, then build the table.
        columns = list(zip(*self.buffer))
        table = pa.table(
            {name: pa.array(col, type=SCHEMA.field(name).type) for name, col in zip(COLUMNS, columns)},
            schema=SCHEMA,
        )
        self.writer.write_table(table)
        self.total += len(self.buffer)
        self.buffer = []

    def close(self) -> None:
        self.flush()
        self.writer.close()
        self.tmp_path.replace(self.out_path)


# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------
def parse_files(
    inputs: list[Path],
    out_path: Path,
    speeds: tuple[str, ...] = ("rapid",),
    player: str | None = None,
    min_moves: int = 0,
    max_games: int | None = None,
    workers: int | None = None,
) -> Counter:
    """Parse every input file into one parquet at out_path. Returns summary counts.

    Can be called from a notebook too:
        parse_files([Path("data/raw/x.jsonl")], Path("data/processed/x.parquet"), player="whopizza1")
    """
    speed_set = {s.lower() for s in speeds}
    stats: Counter = Counter()
    seen_keys: set[str] = set()
    sink = ParquetSink(out_path)
    worker = partial(game_to_rows, player=player, min_moves=min_moves)

    def all_games() -> Iterator[dict]:
        """Chain every input file into one stream, stopping early at --max-games."""
        n = 0
        for path in inputs:
            for info in iter_games(path, speed_set, stats):
                if max_games is not None and n >= max_games:
                    return
                n += 1
                yield info

    def handle(result: tuple[str, str, list[tuple]]) -> None:
        """Tally one finished game and send its rows to the writer (main process only)."""
        status, key, rows = result
        if status != "ok":
            stats[status] += 1
            return
        if key in seen_keys:  # already have this exact game from another file
            stats["skipped_duplicate"] += 1
            return
        seen_keys.add(key)
        stats["games_kept"] += 1
        sink.add(rows)

    workers = workers or os.cpu_count() or 1
    try:
        if workers == 1:
            # Plain loop: easiest to debug (breakpoints and tracebacks just work).
            for info in all_games():
                handle(worker(info))
        else:
            # imap keeps results in input order, so the "first copy wins" duplicate rule
            # is deterministic. chunksize batches games to cut inter-process overhead.
            with Pool(workers) as pool:
                for result in pool.imap(worker, all_games(), chunksize=64):
                    handle(result)
        sink.close()
    except BaseException:
        sink.writer.close()
        sink.tmp_path.unlink(missing_ok=True)  # don't leave a partial file behind
        raise

    stats["positions"] = sink.total
    return stats


def default_out_path(inputs: list[Path], speeds: tuple[str, ...], player: str | None) -> Path:
    """data/processed/<first input name>_<speeds>[_<player>-moves].parquet"""
    name = f"{inputs[0].stem}_{'-'.join(s.lower() for s in speeds)}"
    if player:
        name += f"_{player.lower()}-moves"
    return DEFAULT_OUT_DIR / f"{name}.parquet"


def load_positions(path: str | Path, **read_kwargs):
    """One-liner to get the parsed positions back as a pandas DataFrame.

        from chesspred.data.parse_pgn import load_positions
        df = load_positions("data/processed/chesscom_whopizza1_2026-11-06_rapid_whopizza1-moves.parquet")
    """
    import pandas as pd  # imported here so the parser itself doesn't need pandas loaded
    return pd.read_parquet(path, **read_kwargs)


def print_summary(stats: Counter, out_path: Path, seconds: float) -> None:
    """Human-readable report of what was kept and why things were skipped."""
    speed_counts = {k[6:]: v for k, v in stats.items() if k.startswith("speed_")}
    skips = {k[8:]: v for k, v in stats.items() if k.startswith("skipped_")}
    print(f"\nDone in {seconds:,.1f}s")
    print(f"  games read:        {stats['games_seen']:,}")
    print(f"  by speed:          {dict(sorted(speed_counts.items(), key=lambda kv: -kv[1]))}")
    print(f"  games kept:        {stats['games_kept']:,}")
    if skips:
        print(f"  skipped:           {skips}")
    print(f"  positions (rows):  {stats['positions']:,}")
    if out_path.exists():
        print(f"  file:              {out_path}  ({out_path.stat().st_size / 1e6:,.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path,
                        help=".jsonl (chess.com API), .pgn, or .parquet (Lichess) files; "
                             "list the .jsonl first so its game URLs win when a game is in two files")
    parser.add_argument("--speeds", nargs="+", default=["rapid"],
                        help="time classes to keep (default: rapid). e.g. --speeds rapid blitz")
    parser.add_argument("--player", default=None,
                        help="only keep positions where this username is the one moving")
    parser.add_argument("--min-moves", type=int, default=0,
                        help="drop games shorter than this many full moves (default 0 = keep all)")
    parser.add_argument("--max-games", type=int, default=None,
                        help="stop after this many matching games (quick tests)")
    parser.add_argument("--workers", type=int, default=None,
                        help="parallel processes (default: all CPU cores; 1 = no multiprocessing)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output parquet path (default: data/processed/<input>_<speeds>.parquet)")
    args = parser.parse_args()

    for p in args.inputs:
        if not p.exists():
            parser.error(f"input not found: {p}")
    out_path = (args.out or default_out_path(args.inputs, tuple(args.speeds), args.player)).resolve()

    print(f"Parsing {len(args.inputs)} file(s), keeping speeds {args.speeds}"
          + (f", only {args.player}'s moves" if args.player else "")
          + f"\n  -> {out_path}")
    t0 = time.time()
    stats = parse_files(args.inputs, out_path, tuple(args.speeds), args.player,
                        args.min_moves, args.max_games, args.workers)
    print_summary(stats, out_path, time.time() - t0)


if __name__ == "__main__":
    main()
