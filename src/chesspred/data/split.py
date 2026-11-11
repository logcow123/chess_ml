"""Freeze the train / validation / test split of YOUR games, by date and by game.

Roadmap rules (§8): split by GAME, never by position, and for your games split by DATE.
The newest ~20% of games are the test set that every model (yours and Maia) is scored
on, and that never changes once frozen. The 80% before it is split again into train
and a small validation slice (the newest 10% of all games), so you can tune
hyperparameters and pick checkpoints without ever looking at the test set.

    oldest  |---------------- train (70%) ----------------|-- val (10%) --|---- test (20%) ----|  newest

Usage (from the repo root):
    # Freeze it (one time). Refuses to overwrite an existing split unless you pass --force.
    python -m chesspred.data.split data/processed/chesscom_whopizza1_2026-11-06_rapid_whopizza1-moves.parquet

    # Use it anywhere:
    from chesspred.data.split import add_split_column
    df = add_split_column(df)                      # adds a "split" column
    train = df[df.split == "train"]; test = df[df.split == "test"]

Output: data/splits/whopizza1_rapid.csv (one row per game: game_id, date, split,
n_positions). data/splits/ is NOT gitignored on purpose: commit the file so the split
is frozen in git history too.
"""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from chesspred.data.parse_pgn import REPO_ROOT

SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_SPLIT = SPLITS_DIR / "whopizza1_rapid.csv"
SPLIT_ORDER = ["train", "val", "test"]  # oldest -> newest


def make_split(positions: pd.DataFrame, test_frac: float = 0.2,
               val_frac: float = 0.1) -> pd.DataFrame:
    """Assign every game to train / val / test by date. Returns one row per game.

    Fractions are of GAMES, not positions, so one long game can't tip the balance and
    no game is ever cut in half across two splits.
    """
    # One row per game: when it was played and how many positions it contributes.
    games = (positions.groupby("game_id", as_index=False)
             .agg(date=("date", "min"), n_positions=("fen", "size"))
             # game_id breaks ties so the order is identical on every machine/run.
             .sort_values(["date", "game_id"], ignore_index=True))

    n = len(games)
    n_test = math.ceil(n * test_frac)  # round up: never fewer test games than asked for
    n_val = math.ceil(n * val_frac)
    n_train = n - n_val - n_test
    if n_train <= 0:
        raise SystemExit(f"Only {n} games: not enough for these fractions.")

    # Games are sorted oldest -> newest, so the newest ones land in test.
    games["split"] = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    return games[["game_id", "date", "split", "n_positions"]]


def save_split(games: pd.DataFrame, out_path: Path, source: Path, force: bool = False) -> None:
    """Write the split CSV with a short comment header recording how it was made."""
    if out_path.exists() and not force:
        raise SystemExit(
            f"{out_path} already exists: the split is frozen.\n"
            "Changing it means every result in docs/results.md has to be re-run. "
            "If you really mean it, pass --force.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    with open(out_path, "w") as f:
        # Lines starting with "#" are skipped by load_split (pandas comment="#").
        f.write(f"# Frozen split of your games, made {created}\n")
        f.write(f"# Source: {source.name}\n")
        f.write("# Split by game and by date: oldest = train, then val, newest = test.\n")
        f.write("# DO NOT EDIT. Every model is scored on exactly the 'test' games.\n")
        games.to_csv(f, index=False)


def load_split(path: str | Path = DEFAULT_SPLIT) -> dict[str, str]:
    """Read a frozen split as {game_id: "train" | "val" | "test"}."""
    games = pd.read_csv(path, comment="#")
    return dict(zip(games["game_id"], games["split"]))


def add_split_column(df: pd.DataFrame, path: str | Path = DEFAULT_SPLIT) -> pd.DataFrame:
    """Return df with a "split" column taken from the frozen split file.

    Games that aren't in the file (e.g. games you play AFTER freezing and download
    later) get split="unassigned" and a warning. They are newer than the test set, so
    quietly adding them to train would break the oldest-train / newest-test rule.
    Decide on purpose what to do with them (usually: leave them out, or use them as an
    extra "future" test set).
    """
    mapping = load_split(path)
    out = df.copy()
    out["split"] = out["game_id"].map(mapping).fillna("unassigned")
    n_new = out.loc[out["split"] == "unassigned", "game_id"].nunique()
    if n_new:
        print(f"Warning: {n_new} game(s) are not in the frozen split; marked 'unassigned'.")
    return out


def summarize(games: pd.DataFrame) -> str:
    """Table of games, positions and date range per split (for the printout)."""
    rows = []
    for name in SPLIT_ORDER:
        part = games[games["split"] == name]
        rows.append(f"  {name:<5} {len(part):>5} games  {part['n_positions'].sum():>7,} positions  "
                    f"{part['date'].min()[:10]} -> {part['date'].max()[:10]}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("positions", type=Path,
                        help="parsed parquet from parse_pgn.py (your games)")
    parser.add_argument("--out", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing frozen split (you'll need to re-score everything)")
    args = parser.parse_args()

    positions = pd.read_parquet(args.positions, columns=["game_id", "date", "fen"])
    games = make_split(positions, args.test_frac, args.val_frac)
    save_split(games, args.out, args.positions, args.force)
    print(f"Froze split of {len(games)} games -> {args.out}\n{summarize(games)}")


if __name__ == "__main__":
    main()
