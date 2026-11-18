# Chess Move Predictor

Play against a ~1000-rated bot while a model you trained silently predicts each of your moves, then see how accurate it was.

- Plan: [docs/roadmap.md](docs/roadmap.md)
- Folder guide: [docs/project_structure.md](docs/project_structure.md)
- Results: [docs/results.md](docs/results.md)

## Setup (training)

```bash
pip install -r requirements.txt
pip install -e .
```

## Train the model (RTX 3060)

```bash
# 1. Pre-encode the parsed positions once (writes a .npz next to each parquet)
python -m chesspred.encoding data/processed/lichess_2025-09_blitz-rapid_900-1400_rapid.parquet
python -m chesspred.encoding data/processed/chesscom_whopizza1_2026-11-06_rapid_whopizza1-moves.parquet

# 2. Sanity check, pre-train on Lichess, then fine-tune on your games
python -m chesspred.train configs/resnet_6x64_lichess.yaml --overfit 64
python -m chesspred.train configs/resnet_6x64_lichess.yaml
python -m chesspred.train configs/resnet_6x64_finetune_whopizza1.yaml
```

Each run writes `models/runs/<name>/` (`best.pt`, `last.pt`, `log.csv`, `config.yaml`).
Add `--resume` to continue a stopped run, or `--set model.blocks=10 name=r10x64` to override config values.

## Run the app

```bash
docker compose up   # then open http://localhost:8000
```
