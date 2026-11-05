# Chess Move Predictor — Resources

> **Last updated:** 2026-11-05
> **Companion doc:** [`roadmap.md`](./roadmap.md)
>
> Resources are grouped by what you'll use them for. ⭐ = start here.

---

## 1. Models & research (human move prediction)

These are the closest existing work to this project: models that predict **what a human will play**, not the best move.

| Resource | What it is | Why it matters for this project |
|---|---|---|
| ⭐ [Maia-2 — GitHub (CSSLab/maia2)](https://github.com/CSSLab/maia2) | Pretrained human-move predictor (NeurIPS 2024). One model covers all skill levels: you pass in the FEN plus both players' Elo. `pip install maia2` (Python 3.10–3.12; runs on CUDA, Apple MPS, or CPU). MIT license. | **Your benchmark.** Run it on your own games in week 1 for an instant "how good is state of the art on me?" number. It also has a batch-inference helper that reports accuracy. |
| [Maia-2 paper (arXiv 2409.20553)](https://arxiv.org/abs/2409.20553) · [HTML version](https://arxiv.org/html/2409.20553v2) · [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/250190819ff1dda47cd23cecc0c5a69b-Abstract.html) · local copy: `docs/deep_research/Maia-2_paper.pdf` | The paper behind Maia-2: a 12-block ResNet backbone plus skill-aware attention, ~23.3M params, 18-channel board input, and Elo-bucket embeddings for both players. About 53.3% top-1 vs 51.4% for the best Maia-1. Trained on ~169M Lichess games. | **The blueprint for roadmap models 1.3 and 1.4.** The architecture section tells you exactly how to build a Maia-2-style model. |
| ⭐ [Maia4All: "Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess" (arXiv 2507.21488)](https://arxiv.org/html/2507.21488v1) | Personalizes Maia-2 to one player from **only ~20 games** (previous methods needed ~5,000). Top-1 goes from 51.4% to 53.2% at 20 games. | **The best match to your situation**, since you have few personal games. Roadmap model 2.2; read it before week 3. Check the paper for a code link. |
| [MHR-Maia: "Generative Modeling of Individual Behavior at Scale" (arXiv 2502.14998)](https://arxiv.org/pdf/2502.14998) | Adds low-rank adapters and a per-player "style vector" to Maia. Average move-matching rises from 54.4% to 59.0% across ~48k players using 100 games each, and it can identify a player from their games. | Roadmap model 2.3: personalize with small adapter layers instead of retraining everything. |
| [Chessformer: A Unified Architecture for Chess Modeling (ICLR 2026)](https://mlanthology.org/iclr/2026/monroe2026iclr-chessformer/) · [arXiv HTML](https://arxiv.org/html/2605.19091) · [review summary](https://www.themoonlight.io/en/review/chessformer-a-unified-architecture-for-chess-modeling) | An encoder-only transformer with one token per square, a "Geometric Attention Bias" position encoding, and an attention-based source→destination policy head. This is the architecture behind Maia-3. | The blueprint for roadmap model 1.6 (board transformer). |
| [Maia-2 Rapid on Hugging Face](https://huggingface.co/shermansiu/maia2-rapid) | Hosted weights for the rapid-time-control Maia-2 model. | An alternative way to load the weights. |
| [Maia-3 — GitHub (CSSLab/maia3)](https://github.com/CSSLab/maia3) · [Wiki](https://github.com/CSSLab/maia3/wiki) | Newest Maia: a transformer family ("Chessformer") in 3M, 5M (CPU-friendly), 23M, and 79M (most accurate) sizes. Runs as a UCI engine; weights download from Hugging Face. **AGPL-3.0.** | A possible **human-like bot opponent** (stretch goal), or a second benchmark. Note that the AGPL license matters if you ever deploy publicly. |
| [Maia-3 5M model card (Hugging Face)](https://huggingface.co/UofTCSSLab/Maia3-5M) | Official Maia-3 weights. The Maia-3 family reports **57.1%** move-matching, is conditioned on player rating, and was trained on Lichess games from Jan 2023 to Jul 2025. | The most accurate benchmark available, and small enough to run on a CPU in the app. |
| [Maia-3 ONNX weights](https://huggingface.co/bqrio/maia3-onnx) | A community ONNX export of Maia-3. | ONNX could let you run Maia in the browser or without PyTorch. |
| ⭐ [Maia-1 blog post](https://www.maiachess.com/blog/maia-v1) · [Maia-1 GitHub](https://github.com/CSSLab/maia-chess) | The original Maia: nine models for the 1100–1999 Lichess bands, each trained on 12M games. About 46–52% move-matching accuracy; personalized versions reach up to 65%. | A **readable intro to the whole problem**. It covers good evaluation practice, like dropping the first 10 ply and very-low-clock moves from test sets. |
| [Maia-1 paper: "Aligning Superhuman AI with Human Behavior" (arXiv 2006.01855)](https://arxiv.org/pdf/2006.01855) · [Papers with Code](https://astro.paperswithcode.com/paper/aligning-superhuman-ai-and-human-behavior) | The KDD 2020 paper. | Background for your report: why human-move prediction is different from engine strength. |
| ⭐ [Learning Models of Individual Behavior in Chess (arXiv 2008.10086)](https://arxiv.org/abs/2008.10086) · [HTML version](https://ar5iv.labs.arxiv.org/html/2008.10086) · [Microsoft Research page](https://www.microsoft.com/en-us/research/?p=863784) | Fine-tuning Maia on a single player's games to predict *that player* (and to identify players by style). | **This is exactly your week 3 personalization step.** Read it for how they fine-tune and how many games they needed. |
| [Maia website](https://www.maiachess.com) | You can play against Maia bots online. | A feel for "human-like" bot behavior before you build your own bot. |
| [Grandmaster-Level Chess Without Search — DeepMind (searchless_chess)](https://github.com/google-deepmind/searchless_chess) · [paper](https://br.arxiv.org/pdf/2402.04494) | Transformers (9M / 136M / 270M checkpoints, JAX) trained to imitate Stockfish without any search, plus the ChessBench dataset. Code is Apache 2.0; weights are CC-BY 4.0. | Shows how transformers are applied to board → move prediction and how boards and moves are tokenized. It predicts *engine* moves, not human ones, so read it for architecture ideas (roadmap model 2.6). |
| [Chess-GPT: chess_llm_interpretability (Adam Karvonen)](https://github.com/adamkarvonen/chess_llm_interpretability) · [paper](https://arxiv.org/html/2403.15498v2) · [nanoGPT training fork](https://github.com/adamkarvonen/nanoGPT) · [dataset](https://huggingface.co/datasets/adamkarvonen/chess_games) | A small GPT (nanoGPT, e.g. 8 layers) trained on Lichess games written as PGN text. It predicts the next move from the game so far, and the paper shows it learns an internal board and player-skill estimate. MIT license. | The starting point for roadmap model 1.7 (sequence model over the game's moves). The training code is simple enough to adapt. |
| [Comparing architectures for chess policy networks (Cazenave)](https://lamsade.dauphine.fr/~cazenave/papers/ComparingArchitecturesChess.pdf) | Compares residual, mobile, and other network architectures for chess move prediction. | Useful for choosing ResNet width and depth, and for the "architecture comparison" part of your report. |

---

## 2. Data

| Resource | What it is | Notes |
|---|---|---|
| ⭐ **Your chess.com games**: `GET https://api.chess.com/pub/player/whopizza1/games/archives` then `GET https://api.chess.com/pub/player/whopizza1/games/{YYYY}/{MM}/pgn` | chess.com's free public API (no key needed). The archives endpoint lists your months; the second endpoint returns a month's games as PGN. | **Pull your full history in week 1.** Set a descriptive `User-Agent` header and fetch months one at a time. Python wrapper: [chess.com Python client docs](https://chesscom.readthedocs.io/en/latest/members/chessdotcom.endpoints.player_games_by_month.html). Example script: [gist](https://gist.github.com/andreij/0e3309200c0a6bb26308817a168203f3). |
| ⭐ [Lichess/standard-chess-games on Hugging Face](https://huggingface.co/datasets/Lichess/standard-chess-games) | The official Lichess game database, mirrored as **parquet files partitioned by year and month**. | **The easiest way to get millions of human games.** You can download a single month's shards (or stream them) and filter by Elo and time control with pandas or polars. There's no need to decompress giant PGNs. |
| [Lichess open database (database.lichess.org)](https://database.lichess.org) · [GitHub: lichess-org/database](https://github.com/lichess-org/database) | Monthly dumps of every rated Lichess game as compressed PGN. | The original source. Files are very large; only use this if the HF mirror doesn't fit your needs. Many games include clock (`%clk`) annotations, which allow time-pressure features later. |
| [Icannos/lichess_games (HF)](https://huggingface.co/datasets/Icannos/lichess_games) | A community Hugging Face loader for Lichess games. | An alternative loader. |

**Data tips**
- chess.com and Lichess ratings **aren't on the same scale**, so check the Elo band against your games empirically (roadmap decision #2).
- Drop bullet games and abandoned or very short games.
- Split **by game** (and by **date** for your own games) to avoid leakage.

---

## 3. Python libraries

| Library | Use it for |
|---|---|
| ⭐ [python-chess](https://github.com/niklasf/python-chess) · [docs](https://python-chess.readthedocs.io/en/latest/) | Everything chess-related in Python: parsing PGNs (`chess.pgn`), board state and FEN, legal-move generation (for the illegal-move mask), and driving Stockfish (`chess.engine`). `pip install chess` |
| [PyTorch](https://pytorch.org) | The model and training loop. Install the **CUDA build** on the RTX 3060 machine (pick your CUDA version with the selector on pytorch.org). Colab comes with it preinstalled. |
| [LightGBM](https://lightgbm.readthedocs.io) / [scikit-learn](https://scikit-learn.org) | The classical-ML baseline (roadmap model 0.4): ranking legal moves on hand-made features. |
| [TensorBoard](https://www.tensorflow.org/tensorboard) / [Weights & Biases](https://wandb.ai) | Optional experiment tracking, so runs from the 3060 and Colab can be compared in one place. |
| [pandas](https://pandas.pydata.org) / [polars](https://pola.rs) + [pyarrow](https://arrow.apache.org/docs/python/) | Loading and filtering the parquet datasets. polars is much faster on big files. |
| [Hugging Face `datasets`](https://huggingface.co/docs/datasets) / [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub) | Downloading or streaming individual Lichess parquet shards. |
| [FastAPI](https://fastapi.tiangolo.com) + [Uvicorn](https://www.uvicorn.org) | The app backend (`/predict`, `/bot-move`, game logging). |
| [stockfish (PyPI)](https://pypi.org/project/stockfish) | An alternative Stockfish wrapper with a simpler API than `chess.engine`. Either works. |

---

## 4. Bot opponent (engine)

| Resource | Notes |
|---|---|
| ⭐ [Stockfish](https://stockfishchess.org) · [GitHub](https://github.com/official-stockfish/Stockfish) | The strongest open-source engine. In the Docker image: `apt-get install stockfish`; on a Mac: `brew install stockfish`. |
| [Stockfish UCI options reference](https://github.com/official-stockfish/Stockfish/wiki/UCI-&-Commands) | `UCI_Elo` only goes from **1320 to 3190**, calibrated against engines (CCRL), not chess.com players. `Skill Level` is 0–20, and it works by sometimes picking weaker moves. **For your ~1000 default**, combine a low Skill Level with a depth or time limit, then calibrate by playing (roadmap decision #8). |
| Maia-2 / Maia-3 (see §1) | Stretch goal: a **human-like** bot at a chosen rating instead of a weakened Stockfish. |

---

## 5. Frontend (playable board)

| Resource | Notes |
|---|---|
| ⭐ [react-chessboard](https://github.com/Clariity/react-chessboard) | A drag-and-drop chessboard React component. Pairs with chess.js. |
| ⭐ [chess.js](https://github.com/jhlywa/chess.js) | A JavaScript chess rules engine: legal moves, check and mate detection, and FEN/PGN in the browser. |
| [chessboard.js](https://chessboardjs.com) | Plain-JS board with no React. The **fallback** if you want the simplest possible frontend. |
| [Lichess blog: Chess web programming — game review](https://lichess.org/@/HollowLeaf/blog/chess-web-programming-part-five-game-review/32JsEsrs) | Part of a series on building chess web UIs. The "game review" part is close to your end-of-game report screen. |

---

## 6. Example projects & tutorials to skim

Community projects, so quality varies. Use them for ideas, not as code to copy.

| Project | What to look at |
|---|---|
| [smpurkis/chess_cnn_supervised](https://github.com/smpurkis/chess_cnn_supervised) | Training a CNN on a collection of PGN games (supervised move prediction). Closest to your week 2 pipeline. |
| [colurw/chess_NN](https://github.com/colurw/chess_NN) · [wiki](https://github.com/colurw/chess_NN/wiki) | Neural-network chess project with write-ups on design choices. |
| [Zeta36/Policy-chess](https://github.com/Zeta36/Policy-chess) | Policy network (predicts a move given a position), the same idea as your model's output head. |
| [Medium: "A chess engine was trained on millions of human games… the result: Maia"](https://medium.com/carre4/a-chess-engine-was-trained-on-millions-of-human-games-the-result-maia-89702fd55231) | A plain-language overview of Maia, good for your report intro. |

---

## 7. Packaging the app (Docker)

| Resource | Notes |
|---|---|
| [Docker docs: multi-stage builds](https://docs.docker.com/build/building/multi-stage/) | Build the React frontend in a Node stage, then copy only the built files into the slim Python runtime image. |
| [Docker Compose docs](https://docs.docker.com/compose/) | Start the app with one command (`docker compose up`); mount model weights as a volume. |
| [FastAPI in containers](https://fastapi.tiangolo.com/deployment/docker/) | The official guide to a FastAPI Dockerfile. |
| [FastAPI static files](https://fastapi.tiangolo.com/tutorial/static-files/) | Serve the built frontend from the same container as the API. |
| [PyTorch install selector](https://pytorch.org/get-started/locally/) | Choose **CPU** for the app image (much smaller than the CUDA build) and **CUDA** for the 3060 training machine. |

---

## 8. Training compute

| Resource | Notes |
|---|---|
| [PyTorch automatic mixed precision](https://pytorch.org/docs/stable/amp.html) | `torch.autocast` speeds up training and saves VRAM on the RTX 3060 (bf16 is supported). |
| [Google Colab](https://colab.research.google.com) | Mount Google Drive for data and checkpoints, and save every epoch because sessions can disconnect. |

---

## 9. Suggested reading order

| # | When | Reading | Links | Why |
|---|---|---|---|---|
| 1 | Week 1 (30 min) | **Maia-1 blog post** | [maiachess.com/blog/maia-v1](https://www.maiachess.com/blog/maia-v1) | Understand the problem and how it's evaluated. |
| 2 | Week 1 (1 hr) | **python-chess docs**: PGN parsing, plus `Board` and `legal_moves` | [PGN parsing and writing](https://python-chess.readthedocs.io/en/latest/pgn.html) · [full docs (see "Core")](https://python-chess.readthedocs.io/en/latest/) | You'll use both for the week 1 parser and the legal-move mask. |
| 3 | Week 1 | **Maia-2 README** | [github.com/CSSLab/maia2](https://github.com/CSSLab/maia2) · [PyPI: maia2](https://pypi.org/project/maia2/) | Get it running on your games for the week 1 comparison row. |
| 4 | Before week 2 | **Maia-2 paper**, architecture section | [arXiv HTML](https://arxiv.org/html/2409.20553v2) · [arXiv abstract / PDF](https://arxiv.org/abs/2409.20553) · local copy: `docs/deep_research/Maia-2_paper.pdf` | The blueprint for your ResNet and Elo conditioning. |
| 5a | Before week 3 | **Maia4All**: "Learning to Imitate with Less" | [arXiv HTML](https://arxiv.org/html/2507.21488v1) | Few-game personalization, which fits your ~50-game situation. |
| 5b | Before week 3 | **Learning Models of Individual Behavior in Chess** | [arXiv abstract / PDF](https://arxiv.org/abs/2008.10086) · [HTML version](https://ar5iv.labs.arxiv.org/html/2008.10086) | The original "fine-tune on one player" method. |
| 5c | Optional, week 3 | **MHR-Maia**: "Generative Modeling of Individual Behavior at Scale" | [arXiv PDF](https://arxiv.org/pdf/2502.14998) | Adapter-style personalization (only small added layers are trained). |
| 6a | Before week 4 | **FastAPI in Docker** | [FastAPI: Docker deployment](https://fastapi.tiangolo.com/deployment/docker/) · [FastAPI: static files](https://fastapi.tiangolo.com/tutorial/static-files/) · [Docker: multi-stage builds](https://docs.docker.com/build/building/multi-stage/) | The app container. |
| 6b | Before week 4 | **Stockfish UCI options** | [Stockfish wiki: UCI & Commands](https://github.com/official-stockfish/Stockfish/wiki/UCI-&-Commands) | `Skill Level`, `UCI_Elo`, and depth limits for the ~1000 bot. |
| 7 | Before week 5 | **react-chessboard + chess.js READMEs** | [react-chessboard](https://github.com/Clariity/react-chessboard) · [chess.js](https://github.com/jhlywa/chess.js) | The board UI and move legality in the browser. |
| 8 | Optional | **Chessformer paper** (only if you try the transformer) | [ICLR 2026 page](https://mlanthology.org/iclr/2026/monroe2026iclr-chessformer/) · [arXiv HTML](https://arxiv.org/html/2605.19091) | The architecture behind Maia-3. |
