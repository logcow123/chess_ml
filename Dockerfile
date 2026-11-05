# Multi-stage build (filled in weeks 4-5):
#   Stage 1: Node builds app/frontend
#   Stage 2: slim Python + Stockfish (apt) + CPU-only PyTorch + FastAPI, serves the frontend on :8000
