# Project Structure

What each folder and file is for. The layout follows roadmap §9.

## Top level

| Path | Purpose |
|---|---|
| `README.md` | Setup steps, how to train, how to run the app. Expanded in week 6. |
| `pyproject.toml` | Makes `src/chesspred` installable with `pip install -e .` so the same code runs on the laptop, the 3060 and Colab. |
| `requirements.txt` | Training dependencies (GPU PyTorch, data, notebooks, Maia-2). |
| `requirements-app.txt` | App-only dependencies for Docker (CPU PyTorch, FastAPI, python-chess). |
| `Dockerfile` | Multi-stage build: Node builds the frontend, then a slim Python image with Stockfish and CPU PyTorch serves it. Filled in weeks 4–5. |
| `docker-compose.yml` | One service on port 8000; mounts `models/` so weights stay out of the image. |
| `.dockerignore` | Keeps data, checkpoints, notebooks and docs out of the Docker image. |
| `.gitignore` | Keeps raw/processed data and model checkpoints out of git. |

## `data/`

| Path | Purpose |
|---|---|
| `data/raw/` | Untouched source data: your chess.com PGNs (`chess_com_games_2026-11-04.pgn` lives here now) and Lichess downloads. |
| `data/processed/` | Parsed parquet files and pre-encoded arrays ready for fast training. |

## `docs/`

| Path | Purpose |
|---|---|
| `roadmap.md` | The 6-week plan. |
| `resources.md` | Papers, code and datasets. |
| `cnn_resnet_tutorial.md` | Week 2 walkthrough of encoding, SimpleCNN and ResNet. |
| `results.md` | Running comparison table (§8) and a log of every training run. |
| `project_structure.md` | This file. |
| `deep_research/` | Local copies of papers (Maia-2 PDF, ...). |

## `notebooks/`

EDA (`01_eda.ipynb`), error analysis, and `colab_train.ipynb` for running training on Colab.

## `configs/`

One YAML file per experiment (model type and size, data size, learning rate, seed). `example.yaml` shows the format. Log the config name in `results.md` for every run.

## `src/chesspred/` (the Python package)

| Path | Purpose |
|---|---|
| `data/download_chesscom.py` | Pulls your full game history from the chess.com API. |
| `data/download_lichess.py` | Downloads and filters Lichess games to your Elo band and time controls. |
| `data/parse_pgn.py` | PGN → `(fen, move_uci, player_elo, opponent_elo, time_control, move_number, game_id)` parquet. |
| `encoding.py` | Board → 8×8×18 tensor, move ↔ 4,096 index, legal-move mask. |
| `models/mlp.py` | Model 1.1, the MLP sanity check. |
| `models/cnn.py` | Model 1.2, plain CNN. |
| `models/resnet.py` | Models 1.3/1.4, the ResNet policy network (with optional Elo conditioning). The main model. |
| `models/transformer.py` | Model 1.6, board transformer (stretch). |
| `train.py` | Training loop: split by game, mixed precision, resumable checkpoints, top-1/top-3 logging. |
| `evaluate.py` | Scores any model on the frozen test split: all moves, excluding openings, and by phase. |

## `app/`

| Path | Purpose |
|---|---|
| `app/backend/` | FastAPI server (`main.py`): `/predict`, `/bot-move`, game logging and summary, Stockfish integration, CPU model inference. |
| `app/frontend/` | react-chessboard + chess.js UI: game setup, the board, and the end-of-game accuracy report. |

## `models/`

Trained checkpoints (`v1.pt`, `v2_personal.pt`, ...). Gitignored; keep copies in Google Drive.

## `tests/`

Unit tests (`pytest`). `test_encoding.py` round-trips every move in your PGN; backend tests come in week 4.
