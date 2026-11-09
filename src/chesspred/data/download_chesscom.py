"""Download your full game history from the chess.com public API into data/raw/.

Usage:
    python -m chesspred.data.download_chesscom --username whopizza1
    python -m chesspred.data.download_chesscom --username whopizza1 --months 12

Writes one JSON object per line (one line per monthly archive), newest month first.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Repo root = three levels up from src/chesspred/data/
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "raw"
HEADERS = {"User-Agent": "chess-move-predictor/1.0 (CSE455 class project)"}


def get_json(session, url, retries=3):
    """GET a URL and return parsed JSON, retrying on rate limits and transient errors."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            print(f"  network error ({e}), attempt {attempt}/{retries}")
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:  # rate limited: back off and retry
                print(f"  rate limited, attempt {attempt}/{retries}")
            else:
                print(f"  HTTP {resp.status_code} for {url}")
                if resp.status_code < 500:
                    return None  # 4xx other than 429 won't fix itself
        time.sleep(2 ** attempt)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--username", required=True)
    parser.add_argument("--months", type=int, default=None,
                        help="Only the newest N months (default: full history)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    session = requests.Session()
    session.headers.update(HEADERS)

    archives_url = f"https://api.chess.com/pub/player/{args.username.lower()}/games/archives"
    data = get_json(session, archives_url)
    if not data or not data.get("archives"):
        sys.exit(f"No archives found for {args.username}")

    month_urls = data["archives"][::-1]  # newest first
    if args.months is not None:
        month_urls = month_urls[: args.months]  # slicing never raises IndexError

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"chesscom_{args.username.lower()}_{time.strftime('%Y-%m-%d')}.jsonl"
    tmp_path = out_path.with_suffix(".jsonl.part")

    n_games, failed = 0, []
    with open(tmp_path, "w") as f:
        for i, url in enumerate(month_urls, 1):
            print(f"[{i}/{len(month_urls)}] {url.rsplit('/games/', 1)[-1]}")
            month = get_json(session, url)
            if month is None:
                failed.append(url)
                continue  # never write a `null` line
            f.write(json.dumps(month) + "\n")
            n_games += len(month.get("games", []))
            time.sleep(0.5)  # be polite: requests are serial anyway

    tmp_path.replace(out_path)  # only rename once the whole download finished
    print(f"Saved {n_games} games from {len(month_urls) - len(failed)} months to {out_path}")
    if failed:
        print(f"WARNING: {len(failed)} months failed:", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()