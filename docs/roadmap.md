# Chess Move Predictor — 6-Week Roadmap

> **Version:** v0.4 (first iteration, expect changes)
> **Last updated:** 2026-11-05
> **Project window:** Wed Nov 4 → **Sat Dec 12, 2026 (due date)**, about 5.5 weeks
> **Cadence:** visible progress every week. Each week ends with a deliverable, a `results.md` update, and a changelog line here.
> **Companion doc:** [`resources.md`](./resources.md)

---

## 1. Goal

By the end of week 6, have a **working app where you play against a bot**. On every one of your moves, the model **silently predicts what you will play** from the current board position. When the game ends, the app shows **how accurate the model was**.

### What "done" means

| Piece | Done when… |
|---|---|
| **Model** | A model **you designed and trained yourself**, then **fine-tuned on your games**. That fine-tuned model is the headline result. Given a board (FEN), it returns a ranked list of legal moves with probabilities. |
| **Comparison** | A final table puts your model next to the baselines and the pretrained and fine-tuned Maia models, all on the same test positions (§8). |
| **Accuracy** | It beats simple baselines on held-out games of yours (see §8). First-guess targets, to revisit in week 3, on **all moves including openings**: **≥ 35% top-1** and **≥ 60% top-3**. |
| **App** | It runs **locally in Docker** with one command. You play a full game in the browser against a bot set to **about your rapid rating (~1000 chess.com)**, then get an end-of-game accuracy report. |
| **Report** | The final report explains how much openings inflate accuracy and how your model compares with Maia. |
| **Docs** | The README explains setup, the `results.md` table is filled in, and a short demo exists. |

> **Reference points:** Maia-1 matched human moves about **46–52%** of the time. Maia-2 reaches about **53%**, and Maia-3 reports **57.1%**. Personalized models reach **59%** on average and up to 65–69% for some players. Players at lower ratings are harder to predict, so treat these numbers as a ceiling, not a target.

---

## 2. Starting point (what's in the folder today)

`data/chess_com_games_2026-11-04.pgn`:

- **50 games** as `whopizza1` (Oct 29 → Nov 4, 2026). 26 as White and 24 as Black.
- **~1,590 of your moves**. That's enough to **evaluate and fine-tune** a model but **not enough to train one from scratch**.
- Ratings range from **474 to 1026** (chess.com). Most games are **5-minute** (41/50), plus 7 at 10-minute and 2 at 2+1.
- The PGN has **no clock annotations**, so time-per-move features aren't available from this export.
- Your openings repeat: 1.e4 e5 2.Nf3 as Black (9 games), 1.d4 d5 2.Bf4 as White (7), 1.Nf3 d5 2.e4 (5). **Opening moves will be very easy to predict and will inflate accuracy.** Report them separately (§8).

`docs/deep_research/Maia-2_paper.pdf`: a local copy of the Maia-2 paper.

---

## 3. Strategy in one paragraph

**Design and train your own model**: a general "low-rated human" move predictor trained from scratch on a large sample of public Lichess games at a rating band close to yours. Then **fine-tune it on your own chess.com games** so it learns your habits. **That fine-tuned model is the headline result and the one the app uses.** Pretrained **Maia models are for comparison only**: they're run off the shelf, and Maia-2 is also fine-tuned on exactly the same games of yours, so the comparison is fair (§6, §8). The app runs **locally in a Docker container**, with a bot opponent set to **about 1000 (your chess.com rapid rating)**. Training runs on your **RTX 3060 (12 GB)**, with **Google Colab** for overflow and parallel experiments (§5).

---

## 4. Key design decisions (settle in week 1)

| # | Decision | Default choice | Why / notes |
|---|---|---|---|
| 1 | Whose moves are predicted? | Generic model → fine-tuned on **you** | 1,590 positions is too few to train from scratch; a pretrained model gives a strong base. |
| 2 | Lichess rating band | Start around **900–1400**, then verify | Lichess and chess.com ratings are on different scales. Don't just filter Lichess at 700. One check: run Maia-2 on your games at several Elo inputs and see which Elo predicts you best. |
| 3 | Time controls | Blitz + rapid, **exclude bullet** | Matches what you play; bullet moves are noisy. |
| 4 | Board input | **8×8×18 planes**: 12 piece planes + side to move + 4 castling rights + en passant | Standard (Maia-2 uses this same 18-channel layout). Always encode from the side-to-move's perspective (flip the board for Black). |
| 5 | Move output | **64×64 = 4,096 from→to classes**, with illegal moves masked before softmax | Simple and covers almost every move. Map promotions to queen first; handle underpromotions later if needed. |
| 6 | Model | **Your own** model, trained from scratch in **PyTorch**. Start with a **ResNet-style CNN**, then try the other Tier 1 architectures in §6 | It's the proven architecture, and it trains comfortably on a 3060. Maia models (Tier 2) are used only for comparison and never as the app model. |
| 7 | Training compute | **RTX 3060 12 GB** primary, **Colab** secondary | See §5. |
| 8 | Bot opponent | **Stockfish** via `python-chess`, **default strength ≈ 1000 chess.com rapid** (your rating), adjustable in the UI | Stockfish's `UCI_Elo` setting only goes down to **1320**, and it's calibrated against engines rather than chess.com players. Reaching ~1000 means combining a low `Skill Level` (0–20) with a shallow depth or short think time, then **calibrating by playing it** (week 4). **Backup:** use Maia-3 set to your rating as the bot. That's still fair, because the bot is your opponent, not the predictor. |
| 9 | App stack | **FastAPI** backend + **react-chessboard / chess.js** frontend | Fallback if the frontend eats too much time: a plain HTML page with chessboard.js served by FastAPI. |
| 10 | Deployment | **Local only, in Docker**: one image, started with `docker compose up`, then open `localhost:8000` | Multi-stage build: Node builds the frontend, then a slim Python image runs FastAPI, serves the frontend files, and includes **Stockfish** (installed with `apt`) and **CPU-only PyTorch** with your model weights. Docker is for the app only; training stays outside Docker on the 3060 and in Colab. |

---

## 5. Compute plan

| Resource | Use it for | Notes |
|---|---|---|
| **RTX 3060 (12 GB VRAM)** | Main training runs: the full data pipeline, longer runs, and fine-tuning | Every model in §6 except the very large ones trains comfortably here. Use **mixed precision** (`torch.autocast`; bf16 is supported on this card), `pin_memory=True`, and several DataLoader workers. |
| **Google Colab** | Parallel experiments (such as a hyperparameter sweep while the 3060 runs the main model), and backup if the 3060 is busy | Free-tier sessions disconnect, so **save a checkpoint to Google Drive every epoch** and make training resumable. Keep processed data in Drive as parquet/npz so it's quick to load. |
| **Laptop (CPU) + Docker** | Data parsing, and the app container (model inference during games) | The models here are small enough to predict one position quickly on a CPU, so the container doesn't need GPU support. |

**Setup tips**
- **Keep one codebase** for both machines: push to GitHub and `pip install -e .` on the GPU machine and in Colab.
- **The data loader is usually the bottleneck**, not the GPU. Pre-encode positions into compact arrays once (week 1–2) rather than parsing PGNs or FENs inside the training loop.
- Track runs in `docs/results.md`, plus optionally TensorBoard or Weights & Biases, so results from the 3060 and from Colab end up in one place.

---

## 6. Candidate models to explore

Ordered from simplest to most advanced. ⭐ = recommended core path. The papers and code are linked in [`resources.md`](./resources.md).

### Tier 0: Baselines (no deep learning, week 1)

| # | Model | Idea | Why try it |
|---|---|---|---|
| 0.1 | **Random legal move** | Pick uniformly among legal moves. | The floor. With ~30 legal moves per position, that's about 3% top-1. |
| 0.2 | **Position-frequency lookup** | "In this exact position, what did people (or you) play most often?" | Strong in openings, useless later. It shows how much your repeated openings inflate accuracy. |
| 0.3 | **Stockfish top move** | Predict the engine's best move. | Shows how far human play is from "best" play. A good talking point for the report. |
| 0.4 | **Feature-based move ranker** (logistic regression or LightGBM) | Score each legal move on hand-made features such as capture, check, piece moved, material won, and moves toward the center. Pick the highest score. | A "classical ML" comparison for CSE455, and fully explainable. |

### Tier 1: Build your own neural networks (weeks 2–3) · *your headline model comes from here*

| # | Model | Idea | Fits on 3060? | Notes |
|---|---|---|---|---|
| 1.1 | **MLP** | Flatten the board (768 one-hot inputs) → hidden layers → 4,096 move logits. | Easily | A quick sanity check that the pipeline works. Expect it to lose to CNNs. |
| 1.2 | **Plain CNN** | 4–6 conv layers over the 8×8×18 board → policy head. | Easily | Shows what convolution adds over an MLP. |
| 1.3 ⭐ | **ResNet policy network** (AlphaZero / Leela / Maia style) | 6–12 residual blocks, 64–256 channels, optional squeeze-and-excitation layers, policy head with legal-move mask. | Comfortably | **The main model.** Maia-2's backbone uses 12 conv blocks. Start at 6 blocks × 64 channels and scale up while validation improves. |
| 1.4 ⭐ | **ResNet + Elo conditioning** (Maia-2 style) | Add learned embeddings for player and opponent Elo buckets, mixed into the board features. | Comfortably | One model covers your whole 474–1026 range. This is the main idea behind Maia-2. |
| 1.5 | **ResNet + move history** | Stack the last N board positions as extra input planes (Maia-1 used a six-position history). | Comfortably | Uses "previous game information" directly, and might capture plans or tempo. Compare against 1.3 to see whether history helps. |
| 1.6 | **Board transformer** (Chessformer-lite / ViT style) | Each of the 64 squares is a token → transformer encoder → source-to-destination attention policy head. | Yes, at ~5–25M params | This is the architecture behind Maia-3, the current state of the art. A good "advanced" experiment if 1.3 and 1.4 are done early. |
| 1.7 | **Sequence model over the game's moves** (GPT/nanoGPT or LSTM on PGN text) | Read the game so far as text and predict the next move token. | Small models (~25–50M) yes | A very different approach: it sees the whole game, not just the board. Needs a filter for illegal output and takes longer to train well. Treat it as a stretch goal. |

### Tier 2: Pretrained models (comparison only)

These are the yardstick for your own model. They're run off the shelf, and Maia-2 is also fine-tuned on the same training games as your model. Read their papers for ideas, but **don't use their weights inside your model**.

| # | Model | What it is | Fits on 3060? | How to use it here |
|---|---|---|---|---|
| 2.1 ⭐ | **Maia-2** | ResNet (12 blocks) + skill-aware attention, ~23M params, ~53% top-1. `pip install maia2`, MIT license. | Inference and fine-tuning both comfortable | **Two comparison rows:** (a) off the shelf on your games (week 1), and (b) **fine-tuned on the same training games as your model** (week 3). |
| 2.2 ⭐ | **Maia4All** | Personalization method built on Maia-2 that models an individual from **only ~20 games**. The paper reports 51.4% → 53.2% top-1 at 20 games. | Yes | **The closest match to your situation** (few personal games). If code is released, add it as a comparison row. Either way, its few-game tricks are fair game to **re-implement in your own model's fine-tuning**. |
| 2.3 | **MHR-Maia** ("Generative Modeling of Individual Behavior at Scale") | Maia + low-rank adapters + a per-player "style vector". About 54.4% → 59.0% average move-matching with 100 games per player. | Yes | An idea to borrow: add small adapter layers (LoRA-style) to **your** model and fine-tune only those on your games, so it doesn't overfit. |
| 2.4 | **Maia-3** (Chessformer) | Transformer family (3M / 5M / 23M / 79M params), **57.1%** move-matching, rating-conditioned. AGPL-3.0. | 5M and 23M easily; 79M for inference | **The strongest off-the-shelf comparison row.** Fine-tuning it is optional (stretch). The 5M model runs on a CPU, so it could also be the stretch-goal human-like bot. The AGPL license matters if you deploy publicly. |
| 2.5 | **Maia-1** | Nine separate models for the 1100–1999 Lichess bands (Leela-format weights). | Yes | Historical reference only. Even its lowest band is above your rating, and the tooling is older. Low priority. |
| 2.6 | **DeepMind searchless chess** | Transformers (9M / 136M / 270M) trained to imitate **Stockfish**, in JAX. | 9M yes; the larger ones are impractical | An architecture and tokenization reference. It predicts engine moves, not human ones, and isn't PyTorch. Low priority. |

### Suggested exploration order

| Week | Models |
|---|---|
| 1 | Tier 0 baselines (0.1–0.3) + **Maia-2** and **Maia-3** off-the-shelf comparison rows on your games |
| 2 | 1.1 MLP → 1.2 CNN → **1.3 ResNet**; keep whichever is best on Lichess validation |
| 3 | **1.4 Elo conditioning**, fine-tune *your* model on you (borrowing Maia4All / adapter ideas). Fine-tune **Maia-2** on the same games as a comparison row. 0.4 feature ranker if you want a classical-ML row |
| 4 | **Freeze the app model**: your best **fine-tuned** Tier 1 model on your held-out games |
| Stretch | 1.5 history, 1.6 transformer, 1.7 sequence model, adapters on your model, fine-tuned Maia-3 row |

> **Decided (v0.3–v0.4):** the headline and app model is **your own Tier 1 model, fine-tuned on your games**. Maia models are comparison rows only. Beating them isn't required; showing honestly where your model lands relative to them is the point.

---

## 7. Week-by-week plan

### Week 1 — Foundations & data (Wed Nov 4 → Tue Nov 10)

**Focus:** get the data flowing and get a first accuracy number on the board.

- [x] Set up the repo structure (§9), `requirements.txt`, and git; push to GitHub.
- [ ] **GPU machine setup:** install CUDA-enabled PyTorch on the 3060 machine and confirm `torch.cuda.is_available()` returns True. Make a Colab notebook that mounts Drive and installs the repo.
- [x] Write `download_chesscom.py` to pull your **full** game history from the chess.com public API. More of your games means better fine-tuning and a bigger test set.
- [ ] Download **one month** of Lichess games (the Hugging Face parquet mirror is easiest). Filter to rated blitz/rapid games in your target Elo band.
- [ ] Write the parser: PGN → rows of `(fen, move_uci, player_elo, opponent_elo, time_control, move_number, game_id)`. Save as parquet.
- [ ] **Quick win:** run **Maia-2** (and Maia-3 if setup is easy) off the shelf on your 50 games. That's the first comparison row, and it shows the bar your own model is aiming at.
- [ ] **Freeze the test split now:** pick your held-out games (newest ~20%) and save their game IDs to a file. Every model, yours and Maia, gets scored on exactly these positions.
- [ ] Baselines 0.1–0.3 from §6.
- [ ] Short EDA notebook covering game lengths, openings, rating distribution, and how often positions repeat.

**Deliverables:** `data/processed/*.parquet`, `notebooks/01_eda.ipynb`, and a first baseline table in `docs/results.md`.
**Done when:** you can load positions in one line, the GPU machine trains a dummy model, and you have baseline and Maia-2 accuracy on your games.

---

### Week 2 — First models (Wed Nov 11 → Tue Nov 17)

**Focus:** end-to-end training, then compare simple architectures.

> 📘 **Start with [`cnn_resnet_tutorial.md`](./cnn_resnet_tutorial.md).** It walks through the encoding, the masks, a SimpleCNN, and a ResNet, all tested on your games. Most of this week's tasks extend that code.

- [ ] `encoding.py`: board → tensor, move ↔ index, legal-move mask. **Pre-encode** the dataset into compact arrays for fast loading.
- [ ] **Unit tests:** encode and decode every move in your PGN and check you get the same move back (test castling, en passant, promotion, and Black-perspective flips).
- [ ] `model.py`: MLP, CNN, and ResNet (§6 models 1.1–1.3) sharing one policy-head interface.
- [ ] `train.py`: data loader, cross-entropy loss, train/val split **by game** (not by position), mixed precision, checkpointing (resumable for Colab), and top-1/top-3 logging.
- [ ] Train on **1–5M positions** on the 3060. Run the MLP/CNN comparison on Colab at the same time.

**Deliverables:** `models/v1.pt` (best of 1.1–1.3), with accuracy on Lichess validation data and on your games.
**Done when:** the ResNet beats the frequency-lookup baseline on Lichess validation data.

---

### Week 3 — Scale up & personalize (Wed Nov 18 → Tue Nov 24)

**Focus:** make the model good, then make it *yours*.

- [ ] Scale to **10M+ positions** (keep adding data while validation accuracy improves). Tune learning rate, model width and depth, and batch size.
- [ ] Add **Elo conditioning** (model 1.4).
- [ ] **Fine-tune on your games** with a **time-based split**: oldest ~80% for training, newest ~20% held out for testing. Try freezing the early layers versus full fine-tuning at a low learning rate.
- [ ] Try few-game personalization tricks from **Maia4All** (2.2) and adapter layers from **MHR-Maia** (2.3), re-implemented in **your** model.
- [ ] **Comparison row:** fine-tune **Maia-2** on the exact same training games and score it on the same test positions (model 2.1). Use the same fine-tuning budget (epochs, learning-rate search) you gave your own model.
- [ ] **Error analysis:** accuracy by game phase (opening, middlegame, endgame), by piece moved, and on captures, checks, and quiet moves.
- [ ] **Mid-project checkpoint:** update the targets in §1 and this roadmap based on real numbers.

**Deliverables:** `models/v2_personal.pt` (your model) and an updated `docs/results.md` comparison table (§8).
**Done when:** your fine-tuned model beats your generic model on your held-out games, and the fine-tuned Maia-2 row is filled in.

---

### Week 4 — App backend + Docker (Wed Nov 25 → Tue Dec 1) · *Thanksgiving week (Thu Nov 26): plan a lighter load*

**Focus:** wrap the model and the bot in an API, running in Docker from day one. **Your best fine-tuned Tier 1 model becomes the "app model"**; model improvements continue on Colab or the 3060 only if there's spare time.

- [ ] FastAPI endpoints:
  - `POST /predict` — `{fen}` → top-k moves with probabilities
  - `POST /bot-move` — `{fen, skill}` → Stockfish move
  - `POST /game/{id}/log` — store prediction vs. actual move per turn
  - `GET /game/{id}/summary` — accuracy stats for a finished game
- [ ] Integrate Stockfish through `python-chess`'s `chess.engine` module.
- [ ] **Calibrate the default bot to ~1000:** try a few low `Skill Level` + depth/time settings, play 3–5 games against each, and pick the one that feels like your rapid opponents. Save it as a named preset ("~1000"). If Stockfish still feels wrong (too perfect, then too random), switch to the Maia-3 backup bot (decision #8).
- [ ] **Dockerfile + `docker-compose.yml`** for the backend: slim Python image, `apt install stockfish`, CPU-only PyTorch, model weights copied in or mounted as a volume. A `.dockerignore` keeps data and checkpoints out of the image.
- [ ] Make sure the **prediction is computed before your move is submitted** and logged server-side, so the result can't leak.
- [ ] Load the model for **CPU inference** (export to TorchScript or ONNX if loading is slow).
- [ ] Simple storage: JSON files or SQLite.
- [ ] Backend tests: a scripted full game runs through the API without errors.

**Deliverables:** `app/backend/`, `Dockerfile`, and `docker-compose.yml`.
**Done when:** `docker compose up` starts the backend, and you can play a complete game through the API (scripted or with curl) against the ~1000 preset and get a summary.

---

### Week 5 — Frontend & end-to-end (Wed Dec 2 → Tue Dec 8)

**Focus:** a playable app.

- [ ] Board UI with **react-chessboard** and **chess.js** for client-side move legality.
- [ ] Game setup: pick your color and the bot's strength (**default: the ~1000 preset**).
- [ ] Add the frontend build to the Dockerfile (multi-stage: Node build → files served by FastAPI), so a **single container** runs the whole app.
- [ ] Hide predictions during the game so they don't influence you. Add an optional "reveal" toggle for demos.
- [ ] **End-of-game report:**
  - Top-1 and top-3 accuracy for the game (**all moves, openings included**, matching the headline metric)
  - A small "excluding opening" figure next to it, so the gap is visible
  - A move-by-move table: your move, the predicted move, and the model's confidence
  - Accuracy by game phase (and optionally a chart over move number)
- [ ] **Play 10+ games yourself.** These are brand-new, truly unseen test data. Save them.

**Deliverables:** a working app, end to end, in one Docker container.
**Done when:** someone else can clone the repo, run `docker compose up`, play a game, and read their accuracy report without your help.
**Feature freeze: end of Tue Dec 8.** After this, only fixes, evaluation, and the write-up.

---

### Week 6 — Evaluate, write up, submit (Wed Dec 9 → **Sat Dec 12, due**) · *4 days*

**Focus:** reliability and storytelling. No new features.

- [ ] Edge cases: promotion UI, castling, checkmate, stalemate, draws, resigning, and refreshing mid-game.
- [ ] **Final evaluation** on the games played in the app: run every row of the §8 comparison table on these new games (log the board positions so the Maia models can be scored on them afterward).
- [ ] README: setup steps (`docker compose up`), architecture diagram, and how to run training (3060 and Colab).
- [ ] **Final report** for CSE455, including:
  - The headline: your **fine-tuned model's** top-1 and top-3 on all moves, openings included
  - **An "openings vs. the rest" section**: accuracy with and without the opening, by game phase, and why your repeated openings make early moves easy to predict
  - The §8 comparison table with the Maia rows, and context on training data size
  - What worked, what didn't, and next steps
- [ ] Demo GIF or video.
- [ ] Clean Docker check: delete the image, rebuild from scratch, and confirm it still runs (Thu Dec 10).
- [ ] Update this roadmap with what actually happened (useful for the write-up).

**Deliverables (Sat Dec 12):** final report, demo, and a clean repo that runs in Docker.

---

## 8. Evaluation plan

**Metrics**
- **Top-1 accuracy**: the model's #1 move equals the move you played. This is the headline number.
- **Top-3 / top-5 accuracy**: your move was in the model's top-k.
- (Optional) average log-probability of the move you played.

**Report every result three ways**
1. **All moves, openings included. This is the headline number** (in the app and the report).
2. **Excluding the first ~8 of your moves.** Your openings repeat, so early moves are easy to predict and inflate the score. This goes in the final report to explain the difference.
3. By phase: opening, middlegame, and endgame.

**Test sets**

| Test set | Purpose |
|---|---|
| Lichess validation (held-out games) | Is the generic model learning? |
| Your newest ~20% of chess.com games | Does personalization help? |
| Games played in the app (weeks 5–6) | Final, truly unseen evaluation |

**Final comparison table** (lives in `docs/results.md`; one row per model, columns = top-1 / top-3 on all moves (headline), plus top-1 excluding openings, for each test set):

| Row | Model | Trained by you? | Fine-tuned on your games? |
|---|---|---|---|
| A | Random legal move | n/a | n/a |
| B | Position-frequency lookup | n/a | n/a |
| C | Stockfish top move | No | No |
| D | (Optional) feature-based ranker | Yes | Yes |
| E | **Your model — generic** (Lichess only) | **Yes** | No |
| F | **Your model — personalized** ⭐ headline | **Yes** | **Yes** |
| G | Maia-2 off the shelf | No | No |
| H | **Maia-2 fine-tuned on your games** | No (pretrained) | Yes, same games as F |
| I | Maia-3 off the shelf | No | No |
| J | (Optional) Maia4All, or Maia-3 fine-tuned | No | Yes |

**Must beat**: rows A–C. **Compare honestly against**: rows G–J. Matching Maia-2 would be a strong result for a 6-week from-scratch model, but it isn't required.

**Fairness rules for the comparison**
- Every row is scored on the **exact same test positions** (the frozen split from week 1).
- Fine-tuned rows (F, H, J) use the **same training games** and a similar tuning budget.
- Give Maia models the **same Elo inputs** you give your own model (if yours uses Elo).
- Report the training data size of each model next to its row: Maia-2 saw ~169M games, so context matters when you present the gap.

**Leakage rules**: split by **game**, never by position. For your games, split by **date**. Never tune hyperparameters on the final test set.

---

## 9. Proposed repo structure

```
chess/
├── data/
│   ├── raw/                 # PGNs (move chess_com_games_*.pgn here), Lichess downloads
│   └── processed/           # parquet / pre-encoded arrays
├── docs/
│   ├── roadmap.md           # this file
│   ├── resources.md
│   ├── results.md           # running table of experiments & accuracy
│   └── deep_research/       # papers (Maia-2 PDF, ...)
├── notebooks/               # EDA, error analysis, colab_train.ipynb
├── configs/                 # one YAML per experiment (model, data size, lr)
├── src/chesspred/
│   ├── data/                # download_chesscom.py, download_lichess.py, parse_pgn.py
│   ├── encoding.py          # board → tensor, move ↔ index
│   ├── models/              # mlp.py, cnn.py, resnet.py, transformer.py
│   ├── train.py
│   └── evaluate.py
├── app/
│   ├── backend/             # FastAPI + Stockfish + model inference
│   └── frontend/            # react-chessboard + chess.js
├── models/                  # checkpoints (gitignored; copies in Google Drive)
├── tests/
├── Dockerfile               # multi-stage: build frontend → slim Python + Stockfish + CPU torch
├── docker-compose.yml       # one service, port 8000, optional volume for model weights
├── .dockerignore            # keep data/ and large checkpoints out of the image
├── requirements.txt         # training deps (GPU)
├── requirements-app.txt     # app deps (CPU-only torch, FastAPI, python-chess)
└── README.md
```

---

## 10. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Lichess files are huge (many GB per month) | High | Use the Hugging Face parquet mirror and stream or filter a single month. You only need tens of millions of positions at most. |
| The data loader is too slow to keep the 3060 busy | Medium | Pre-encode positions into compact arrays once. Use several DataLoader workers and `pin_memory`. |
| Colab sessions disconnect or GPU time runs out | Medium | Save resumable checkpoints to Drive every epoch. Run long jobs on the 3060 and short experiments on Colab. |
| Results drift between the 3060 and Colab (different library versions) | Low | Pin versions in `requirements.txt`, fix random seeds, and log the config with every run. |
| Low-rated play is noisy, so accuracy looks low | High | Set expectations early, report top-3, and compare against the Maia models rather than an absolute number. |
| Only 50 personal games | High | Pull your full chess.com history in week 1, borrow few-game personalization ideas from Maia4All, and save every game played in the app. |
| Too many candidate models, not enough time | Medium | Follow the exploration order in §6. The ⭐ path is the plan; everything else is optional. |
| Frontend takes longer than expected | Medium | Fall back to plain HTML with chessboard.js. The model and the report matter more than the UI. |
| Rating scales don't match (chess.com vs Lichess) | Medium | Check the rating band empirically in week 1 (decision #2). |
| Stockfish can't be made to feel like a ~1000 player | Medium | Its Elo setting only goes down to 1320, so use Skill Level plus depth limits and calibrate by playing. Backup: Maia-3 bot set to your rating (decision #8). |
| The Docker image is huge or slow to build (PyTorch is large) | Medium | Use CPU-only PyTorch wheels, a slim base image, a `.dockerignore`, and keep weights out of the image (mount them). |
| Docker on Apple Silicon (arm64) | Low | Use multi-arch base images. Debian's `stockfish` and CPU PyTorch both have arm64 builds. Test the image on your Mac in week 4, not week 6. |
| Due date (Sat Dec 12) is earlier than the original 6-week window | High | Feature freeze is Tue Dec 8 and week 6 is evaluation and write-up only. Each week has a deliverable, so slips show up early. |

---

## 11. Stretch goals (only after week 5 is done)

- Live "predictability meter" during the game (opt-in).
- **Maia-3 (5M) as a selectable bot** (if it isn't already the backup default), so the opponent plays like a human at a chosen rating.
- A side-by-side in the end-of-game report: your model's prediction next to Maia-2's on each move ("who predicted you better this game?").
- §6 models 1.5 (move history), 1.6 (board transformer), 1.7 (sequence model), and 2.3 (adapters).
- Predict *blunders*: how likely you are to make a big mistake in a position (using Stockfish evaluations).
- *(Not planned. The app is local-only in Docker.)* Public hosting would be possible later, since the same container could run on a host. Check model licenses first; Maia-3 is AGPL.

---

## 12. Decisions log & open questions

### Decided

| Question | Decision | Since |
|---|---|---|
| Due date & cadence | **Due Sat Dec 12, 2026.** Visible progress every week (weekly deliverable + `results.md` + changelog). Feature freeze Tue Dec 8. | v0.4 |
| Headline model must be one you built? | **Yes.** Your own Tier 1 model; Maia models are comparison rows only. | v0.3 |
| Generic or fine-tuned as the headline? | **Fine-tuned** on your games (row F in §8). | v0.4 |
| Where does the app run? | **Locally, inside a Docker container** (decision #10). | v0.4 |
| Default bot strength | **~1000, your chess.com rapid rating.** Calibrated in week 4 (decision #8). | v0.4 |
| Openings in the headline accuracy? | **Included.** The final report explains the with/without-openings gap. | v0.4 |

### Still open

- [ ] What format does CSE455 expect for the final report (paper, slides, live demo)? Plan week 6 around it.
- [ ] What should the end-of-game report count as "the opening": a fixed first ~8 moves, or until your moves leave a known opening line?
- [ ] Is there a minimum number of app games you want to play for the final evaluation (suggested: 10+)?

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| v0.4 | 2026-11-05 | Answered the open questions: due Sat Dec 12 (week 6 shortened to 4 days, feature freeze Tue Dec 8, weekly progress cadence), the fine-tuned model is the headline, the app runs locally in Docker (decision #10, Dockerfile in weeks 4–5, a clean rebuild check in week 6), the default bot is ~1000 with a Stockfish calibration plan and a Maia-3 backup, and openings are included in the headline with the gap explained in the final report. Turned §12 into a decisions log. |
| v0.3 | 2026-11-05 | Decided: your own from-scratch model is the headline and app model; Maia models (off the shelf and fine-tuned) are comparison only. Added a final comparison table and fairness rules (§8), froze the test split in week 1, and updated weeks 3, 4 and 6 and the Tier 2 notes. |
| v0.2 | 2026-11-05 | Training moved off the laptop to an RTX 3060 (12 GB) + Google Colab: added §5 Compute plan and updated weeks 1–4 and the risks. Added §6 Candidate models (baselines, own architectures, pretrained Maia models) with an exploration order. Renumbered later sections. |
| v0.1 | 2026-11-04 | First draft. |
