"""Download one month of Lichess games and keep only the ones this project needs.

Source: the official Lichess parquet mirror on Hugging Face
(https://huggingface.co/datasets/Lichess/standard-chess-games), laid out as
    data/year=YYYY/month=MM/train-00000-of-00NNN.parquet

A recent month is split into hundreds of shards, far more than you need. This script
downloads a few shards at a time, filters each one, keeps the small filtered result and
deletes the big raw shard, so disk use stays low.

Filters (roadmap decisions #2 and #3):
  * rated Blitz or Rapid games only (no bullet, no classical, no casual)
  * BOTH players' Lichess ratings inside the Elo band (default 900-1400)
  * game ended normally or on time (drops abandoned games and rule violations)
  * at least 10 full moves

Usage (from the repo root, after `pip install -e .`):
    python -m chesspred.data.download_lichess --list-months
    python -m chesspred.data.download_lichess --year 2025 --month 6 --shards 2   # quick test
    python -m chesspred.data.download_lichess --year 2025 --month 6 --shards 8   # real run

Output:
    data/raw/lichess/2025-06/shard_00012.parquet  (one filtered file per shard; reruns skip these)
    data/raw/lichess/lichess_2025-06_blitz-rapid_900-1400.parquet  (all shards combined)

Requires: pip install huggingface_hub polars
"""

from __future__ import annotations

import argparse
import random
import re
import time
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "Lichess/standard-chess-games"

# Columns kept from the raw data. `Site` (the game URL) doubles as a unique game id.
COLUMNS = [
    "Site", "Event", "White", "Black", "Result", "WhiteElo", "BlackElo",
    "UTCDate", "TimeControl", "Termination", "ECO", "Opening", "movetext",
]

KEEP_TERMINATIONS = ["Normal", "Time forfeit"]
MIN_FULL_MOVES = 10


# --------------------------------------------------------------------------------------
# Filtering (pure function, easy to test without downloading anything)
# --------------------------------------------------------------------------------------
def filter_games(
    lf: pl.LazyFrame,
    min_elo: int = 900,
    max_elo: int = 1400,
    speeds: tuple[str, ...] = ("Blitz", "Rapid"),
    min_full_moves: int = MIN_FULL_MOVES,
) -> pl.LazyFrame:
    """Keep rated games of the chosen speeds with both players inside the Elo band."""
    # Event looks like "Rated Blitz game" or "Rated Blitz tournament https://...".
    speed_regex = rf"^Rated ({'|'.join(speeds)}) "
    # White's move numbers look like "12. " ("12... " is Black's and doesn't match).
    move_number_regex = r"\d+\. "

    return (
        lf.select(COLUMNS)
        .filter(pl.col("Event").str.contains(speed_regex))
        .filter(
            pl.col("WhiteElo").is_between(min_elo, max_elo)
            & pl.col("BlackElo").is_between(min_elo, max_elo)
        )
        .filter(pl.col("Termination").is_in(KEEP_TERMINATIONS))
        .with_columns(num_moves=pl.col("movetext").str.count_matches(move_number_regex))
        .filter(pl.col("num_moves") >= min_full_moves)
        .with_columns(
            speed=pl.col("Event").str.extract(speed_regex, 1),
            month=pl.col("UTCDate").cast(pl.Utf8).str.slice(0, 7),
        )
    )


# --------------------------------------------------------------------------------------
# Hugging Face helpers
# --------------------------------------------------------------------------------------
def list_months(api: HfApi) -> list[str]:
    """Return every available month as 'YYYY-MM'."""
    months = []
    for year_dir in api.list_repo_tree(REPO_ID, path_in_repo="data", repo_type="dataset"):
        year_match = re.search(r"year=(\d{4})$", year_dir.path)
        if not year_match:
            continue
        for month_dir in api.list_repo_tree(REPO_ID, path_in_repo=year_dir.path, repo_type="dataset"):
            month_match = re.search(r"month=(\d{2})$", month_dir.path)
            if month_match:
                months.append(f"{year_match.group(1)}-{month_match.group(1)}")
    return sorted(months)


def list_shards(api: HfApi, year: int, month: int) -> list[str]:
    folder = f"data/year={year}/month={month:02d}"
    try:
        entries = api.list_repo_tree(REPO_ID, path_in_repo=folder, repo_type="dataset")
        shards = sorted(e.path for e in entries if e.path.endswith(".parquet"))
    except Exception as err:  # folder missing -> nicer message below
        raise SystemExit(
            f"Couldn't list {folder} ({err}). Run with --list-months to see what exists."
        )
    if not shards:
        raise SystemExit(f"No parquet files in {folder}. Run with --list-months.")
    return shards


def process_shard(raw_path: Path, out_path: Path, args) -> tuple[int, int]:
    """Filter one raw shard into out_path. Returns (games_in_shard, games_kept)."""
    total = pl.scan_parquet(raw_path).select(pl.len()).collect().item()
    kept = filter_games(
        pl.scan_parquet(raw_path), args.min_elo, args.max_elo, tuple(args.speeds)
    ).collect()
    kept.write_parquet(out_path)
    return total, kept.height


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-months", action="store_true", help="print available months and exit")
    parser.add_argument("--year", type=int)
    parser.add_argument("--month", type=int)
    parser.add_argument("--shards", type=int, default=4,
                        help="how many shards of the month to use (default 4); 0 = the whole month")
    parser.add_argument("--min-elo", type=int, default=900)
    parser.add_argument("--max-elo", type=int, default=1400)
    parser.add_argument("--speeds", nargs="+", default=["Blitz", "Rapid"])
    parser.add_argument("--seed", type=int, default=0, help="which shards get picked")
    parser.add_argument("--out", type=Path, default=Path("data/raw/lichess"))
    parser.add_argument("--keep-raw", action="store_true", help="don't delete raw shards after filtering")
    args = parser.parse_args()

    api = HfApi()

    if args.list_months:
        months = list_months(api)
        print(f"{len(months)} months available: {months[0]} ... {months[-1]}")
        print("Most recent 12:", ", ".join(months[-12:]))
        return

    if args.year is None or args.month is None:
        parser.error("--year and --month are required (or use --list-months)")
    if (args.year, args.month) == (2024, 10):
        print("Warning: the 2024-10 mirror has a known movetext bug (moves start with '0...'). "
              "Consider another month.")

    tag = f"{args.year}-{args.month:02d}"
    shard_dir = args.out / tag
    raw_dir = args.out / "_raw_tmp"
    shard_dir.mkdir(parents=True, exist_ok=True)

    all_shards = list_shards(api, args.year, args.month)
    if args.shards and args.shards < len(all_shards):
        # Spread picks across the month instead of taking only the first few days.
        chosen = sorted(random.Random(args.seed).sample(all_shards, args.shards))
    else:
        chosen = all_shards
    print(f"{tag}: {len(all_shards)} shards available, using {len(chosen)}")

    grand_total = grand_kept = 0
    for i, remote in enumerate(chosen, 1):
        shard_num = re.search(r"train-(\d+)-of", remote).group(1)
        out_path = shard_dir / f"shard_{shard_num}.parquet"
        if out_path.exists():
            print(f"[{i}/{len(chosen)}] {remote} already done, skipping")
            continue

        t0 = time.time()
        print(f"[{i}/{len(chosen)}] downloading {remote} ...", flush=True)
        raw_path = Path(hf_hub_download(REPO_ID, remote, repo_type="dataset", local_dir=raw_dir))
        size_mb = raw_path.stat().st_size / 1e6
        total, kept = process_shard(raw_path, out_path, args)
        grand_total += total
        grand_kept += kept
        if not args.keep_raw:
            raw_path.unlink()
        print(f"    {size_mb:,.0f} MB raw -> kept {kept:,} of {total:,} games "
              f"({kept / max(total, 1):.1%}) in {time.time() - t0:,.0f}s")

    # Combine every filtered shard for this month into one file.
    parts = sorted(shard_dir.glob("shard_*.parquet"))
    combined = pl.concat([pl.read_parquet(p) for p in parts]).unique(subset="Site")
    final_path = args.out / f"lichess_{tag}_{'-'.join(s.lower() for s in args.speeds)}_{args.min_elo}-{args.max_elo}.parquet"
    combined.write_parquet(final_path)

    positions = int(combined["num_moves"].sum() * 2)  # ~2 plies per full move
    print("\nDone.")
    print(f"  games:              {combined.height:,}")
    print(f"  approx. positions:  {positions:,}  (roadmap week 2 wants 1-5M, week 3 wants 10M+)")
    print(f"  by speed:           {dict(combined['speed'].value_counts().iter_rows())}")
    print(f"  file:               {final_path}  ({final_path.stat().st_size / 1e6:,.0f} MB)")


if __name__ == "__main__":
    main()
