# Maia-2 vs. My Validation Games

Nov 18, 2026 · @Logan Cowley

## Summary

Off-the-shelf Maia-2 (rapid) guesses my move first try 49.5% of the time and has it in its top 3 76.6% of the time, across 2,459 moves from my 75 validation games.

- **Openings are Maia's weakest phase (39.0% top-1), not its easiest.** It expects the typical club player's moves, and my repertoire differs. Dropping my first 8 moves raises top-1 to 52.9%.
- **Endgames are its strongest phase (54.3% top-1, 83.6% top-3).**
- **A higher Elo input doesn't help.** Shifting my rating up by 200–600 points leaves top-1 at 49.4–49.5%, so my chess.com rating goes in as-is.
- **For scale:** a random legal move scores 6.7% top-1 on the same positions. Maia already clears the roadmap's targets for my own model (at least 35% top-1 and 60% top-3).

These are validation numbers. Row G in the results table is still to be scored on the 150 test games.

## Setup

Maia-2's pretrained rapid model was run unchanged on every position where it was my turn, and scored against the move I actually played.

- **Model:** Maia-2 rapid (`maia2` 0.11.0, `models/maia2/rapid_model.pt`), no fine-tuning. Elo inputs were my chess.com rating and my opponent's, as recorded in each game.
- **Data:** my chess.com rapid games as `whopizza1`, parsed to one row per move I played.
- **Split:** the frozen date-based split in `data/splits/whopizza1_rapid.csv`, by game, oldest to newest.

| Split | Games | My moves | Dates |
| --- | --- | --- | --- |
| Train | 521 | 18,208 | Nov 25, 2023 – Jun 26, 2026 |
| **Validation (this report)** | **75** | **2,459** | **Jun 26 – Jul 9, 2026** |
| Test (untouched) | 150 | 5,668 | Jul 11 – Nov 6, 2026 |

In the validation games I played White for 1,234 moves and Black for 1,225. My rating ranged from 950 to 1071 and my opponents' from 944 to 1080.

**Metrics** (roadmap §8):

- **Top-k:** my move was among Maia's k most likely moves (k = 1, 3, 5). Top-1 is the headline.
- **Avg p(my move):** the average probability Maia put on the move I played.
- **Random top-1:** what picking a random legal move would score on the same positions.

**Phases:** opening is my moves 1–8, matching the "excluding openings" cut. Endgame is 6 or fewer queens, rooks, bishops and knights left on the board, roughly the rule Lichess uses. Middlegame is everything else. That gives 600 opening, 1,014 middlegame and 845 endgame moves.

## Results

Maia-2 scores 49.5% top-1 on all of my validation moves, from 39.0% in the opening to 54.3% in the endgame.

| Slice | Moves | Top-1 | Top-3 | Top-5 | Avg p(my move) | Median rank of my move | Random top-1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **All moves (headline)** | **2,459** | **49.5%** | **76.6%** | **86.9%** | **35.5%** | **2** | **6.7%** |
| Excluding my first 8 moves | 1,859 | 52.9% | 80.3% | 89.1% | 38.3% | 1 | 7.6% |
| Opening | 600 | 39.0% | 65.3% | 80.2% | 26.8% | 2 | 4.2% |
| Middlegame | 1,014 | 51.7% | 77.5% | 87.0% | 36.4% | 1 | 5.6% |
| Endgame | 845 | 54.3% | 83.6% | 91.7% | 40.6% | 1 | 9.9% |
| As White | 1,234 | 47.2% | 75.7% | 88.2% | 34.4% | 2 | 5.5% |
| As Black | 1,225 | 51.8% | 77.6% | 85.6% | 36.6% | 1 | 8.0% |

Game to game, top-1 varies a lot: the median game is 50.0%, the middle half of games fall between 44.0% and 56.1%, and the range runs from 20.8% to 75.0%.

## Accuracy by move number

Maia-2 struggles most with my first 4 moves and does best in long games, peaking at 61% top-1 after move 40.

&#91;embedded content: notebooks/maia2\_val\_comparison.ipynb · 2,459 moves from 75 validation games, Jun 26 – Jul 9, 2026\]

The early dip comes from my opening choices (next section). Black's first 4 moves are easier to predict than White's: 35% vs. 28% top-1. All 323 moves after move 40 are endgame positions, with 17.6 legal moves on average vs. 28.5 overall, which gives Maia fewer options to choose from.

## Where Maia gets me wrong

Maia's top guess missed on 1,242 of my moves, but my move was still in its top 3 on 54% of those misses. On 301 misses, Maia was at least 50% sure of a different move.

**My opening repertoire accounts for the most repeated misses.** All ten of the most frequent disagreements come in my first 8 moves, and 366 of the 1,242 misses (29%) are opening moves. As White I play a London-style d4 and Bf4 setup where Maia expects Nf3 and Qd3. As Black I play ...c6 and ...d6 where it expects ...Nf6.

| Colour | Maia expected | I played | Times |
| --- | --- | --- | --- |
| White | Nf3 | d4 | 38 |
| White | Qd3 | Bf4 | 27 |
| Black | Nf6 | c6 | 18 |
| White | c3 | Nf3 | 17 |
| Black | Nf6 | d6 | 14 |
| White | c3 | Bd3 | 14 |
| Black | Qd7 | Nf6 | 10 |
| Black | Qa5+ | d5 | 6 |
| White | Qd2 | Bf4 | 6 |
| Black | Bg4 | g6 | 6 |

**Maia's most confident misses** are useful examples for the report. Twice it was 99% sure I'd recapture hxg3 and I played fxg3 instead. Others are tactical moments it expected me to take, like promoting with ...d1=Q.

| Date | Move | Colour | Phase | I played (Maia's p) | Maia expected (p) | Position |
| --- | --- | --- | --- | --- | --- | --- |
| Jun 26, 2026 | 8 | White | Opening | fxg3 (1.1%) | hxg3 (98.7%) | [Lichess](https://lichess.org/analysis/rn1qk2r/ppp2ppp/4pn2/3p4/3P4/3QPNb1/PPP2PPP/RN2K2R_w_KQkq_-_0_8) |
| Jul 4, 2026 | 5 | White | Opening | fxg3 (1.1%) | hxg3 (98.6%) | [Lichess](https://lichess.org/analysis/rnbqk1nr/ppp2ppp/4p3/3p4/3P4/4P1b1/PPP2PPP/RN1QKBNR_w_KQkq_-_0_5) |
| Jul 9, 2026 | 43 | Black | Endgame | Rd3 (0.6%) | d1=Q (97.0%) | [Lichess](https://lichess.org/analysis/8/8/4p2P/5pR1/4k3/4r3/3p2PK/8_b_-_-_0_43) |
| Jun 27, 2026 | 13 | White | Middlegame | g3 (0.5%) | Nxg8+ (94.9%) | [Lichess](https://lichess.org/analysis/r2q1br1/ppp1kp1p/4pN2/3pP3/5P2/3Q4/PPP2PPP/R3K2R_w_KQ_-_1_13) |
| Jul 8, 2026 | 57 | White | Endgame | Rb8 (1.2%) | b7 (93.5%) | [Lichess](https://lichess.org/analysis/R7/8/1P6/2k5/6Pp/8/6PK/8_w_-_-_0_57) |
| Jul 3, 2026 | 21 | White | Middlegame | Na6 (5.2%) | Nd5 (92.7%) | [Lichess](https://lichess.org/analysis/1rb2r1k/3p2p1/7p/p7/1Np5/4P3/P1B2PPP/R4RK1_w_-_-_0_21) |
| Jul 1, 2026 | 11 | Black | Middlegame | Be6 (0.7%) | Bxh1 (92.1%) | [Lichess](https://lichess.org/analysis/r2qkb1r/ppp2p1p/5p2/3bp3/8/3P2PB/PPP2P1P/R1BQK2R_b_KQkq_-_1_11) |
| Jul 3, 2026 | 25 | White | Endgame | Rxd5 (6.3%) | Bxe8 (91.7%) | [Lichess](https://lichess.org/analysis/4r2k/6p1/b6p/p2p4/B1p5/4PP2/Pr4PP/R2R2K1_w_-_-_1_25) |

## Elo input test

Telling Maia I'm rated higher doesn't improve its first guess. Top-1 stays at 49.4–49.5% up to +600 points and then drops, so my chess.com rating goes in unchanged.

Maia-2 was trained on Lichess ratings, which run on a different scale from chess.com's (roadmap §4, decision 2). It also puts every rating under 1100 into one bucket. To test this, I re-ran the validation games with both my rating and my opponent's shifted up by the same amount, so the gap between us stayed the same.

| Elo shift | My median Elo as Maia sees it | Top-1 | Top-3 | Top-5 | Top-1 excl. first 8 moves | Avg p(my move) |
| --- | --- | --- | --- | --- | --- | --- |
| **+0 (as-is)** | **1014** | **49.5%** | **76.6%** | **86.9%** | **52.9%** | **35.5%** |
| +200 | 1214 | 49.5% | 77.8% | 87.8% | 53.5% | 37.9% |
| +400 | 1414 | 49.5% | 78.5% | 88.5% | 53.1% | 38.8% |
| +600 | 1614 | 49.4% | 78.2% | 89.1% | 52.8% | 39.3% |
| +800 | 1814 | 48.8% | 78.6% | 88.7% | 52.0% | 39.5% |
| +1000 | 2014 | 48.4% | 77.8% | 86.5% | 51.7% | 39.3% |

Higher inputs make Maia slightly more confident in my actual move: avg p goes from 35.5% to about 39%, and top-3 rises by 1–2 points. With about 2,500 positions from 75 games, differences under 1–2 points are within noise.

## Caveats and how to reproduce

- **These are validation numbers, not the final comparison.** Row G (Maia-2 off the shelf) in `docs/results.md` must be scored on the 150 test games, the same positions every other model gets.
- **Validation is small and covers two weeks.** 75 games from Jun 26 to Jul 9, 2026, at ratings 950–1071, so differences under 1–2 points are noise.
- **The phase boundaries are my choice.** The roadmap doesn't define them. The cut-offs are listed under Setup.
- **The roadmap puts Maia-2 at about 53% top-1 on Lichess players**, and treats that as a ceiling. My 49.5% is close to that, even though all my ratings fall in Maia-2's lowest bucket (under 1100).

To reproduce, run `notebooks/maia2_val_comparison.ipynb` from the repo. It takes about a minute on the CPU, including the Elo test. Per-move predictions are saved to `data/processed/maia2_rapid_val_predictions.parquet`.
