# Bengali Multi-Label Cyberbullying Detection — Implementation Plan (v5 Corrected)

This document describes the **current** implementation in
`bengali-cyberbullying-lightweight-v4.ipynb`, the problems it fixes relative to the previous
version, the model architecture (~10M parameters), and how to run it on **Kaggle with 2x T4 GPUs**.

---

## 1. Problem & data

- **Task:** multi-label classification of Bengali social-media comments.
- **Toxic labels predicted by the model (4):** `vulgar`, `threat`, `troll`, `insult`.
- **`neutral`** is **not** predicted — it is derived deterministically as `NOT(any toxic)`.
- **Dataset:** `combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral_plus_threat300.csv`
  (~15.5k rows). After cleaning + dedup on cleaned text: ~15.3k rows.

Class rates (full, post-clean): vulgar ~24.9%, threat ~14.0%, troll ~28.4%, insult ~28.5%,
neutral ~35.7%.

---

## 2. What was wrong before and how it is fixed

| # | Previous problem | Impact | Fix in this notebook |
|---|------------------|--------|----------------------|
| 1 | **Augmentation ran before the split** (whole dataframe), then split. | Augmented near-duplicate threat rows leaked across train/val/test → **inflated `threat` F1 (~0.88)**. | Split **first**; augment **train only**; dedup on **cleaned** text; hard assertion `train∩val = train∩test = 0`. |
| 2 | **Broken "mixup"** kept one sample's tokens but blended another sample's labels. | Injected label noise into ~50% of batches → **low precision** (troll 0.51, insult 0.66). | Mixup removed entirely. Train F1 now computed on real labels. |
| 3 | **Stacked imbalance tricks**: `pos_weight≈3x` + Focal + low tuned thresholds. | Heavy over-prediction; macro precision 0.73 vs recall 0.85. | Focal Loss **without** `pos_weight`; class balance comes from train-only augmentation; thresholds tuned in a safe `[0.30, 0.70]` band. |
| 4 | **`neutral` modeled as an independent sigmoid** despite being deterministic. | Possible `neutral`+toxic contradictions; wasted capacity. | Head outputs 4 toxic logits; neutral derived. Predictions are contradiction-free by construction (asserted). |
| 5 | **SWA built but never used** at eval; no BatchNorm anyway. | Dead code / wasted compute. | SWA removed. |
| 6 | Model was ~3.5M params (title claimed 8.2M). | Mismatch with the stated budget. | Rebuilt to **~9.74M** params (verified), still uses **T4x2 DataParallel**. |
| 7 | Metrics were partly leakage-inflated and partly precision-starved. | Misleading headline number. | Honest evaluation: neutral derived, all 5 classes reported, no leakage. |

> **Expected effect on numbers:** the `threat` score will look more modest than the previous
> ~0.88 (that was partly leakage), while `troll`/`insult` **precision** should improve because
> the label-noise (mixup) and over-prediction (stacked weighting) are gone. The headline
> macro-F1 becomes trustworthy rather than optimistic.

---

## 3. Pipeline (cell by cell)

1. **Setup & imports** — seeds, device, GPU detection (prints each T4).
2. **Config** — all hyperparameters (see §4 & §5).
3. **Load + clean + dedup + derive neutral** — clean text (URLs/mentions/emoji/punct stripped,
   Bengali range kept), drop `< MIN_WORDS`, **dedup on `clean_text`**, set `neutral = NOT(any toxic)`.
4. *(markdown)* why we split before augmenting.
5. **Stratified split BEFORE augmentation** — `MultilabelStratifiedShuffleSplit` 70/15/15;
   records `VAL_TEST_TEXTS` for the leakage guard.
6. *(markdown)* train-only augmentation rationale.
7. **Threat augmentation (train only)** — swap / deletion / pseudo-synonym; drops any augmented
   text colliding with val/test; **asserts zero leakage** and **neutral consistency**.
8. **EDA** — train per-class counts, toxic-labels-per-comment, token-length distribution.
9. **Vocab build (train only)** — word vocab (`MIN_FREQ=2`, cap `30000`) + char vocab; prints val OOV.
10. **FastText `cc.bn.300`** — streamed, only vocab words kept; OOV words get scaled random init;
    prints coverage.
11. **Dataset & DataLoaders** — targets are the **4 toxic** labels; word-dropout on train.
12. **Model build** — prints param count and the `~10M` check; wraps in `nn.DataParallel` if `>1` GPU.
13. **Loss + optimizer + scheduler** — Focal (no `pos_weight` by default); AdamW; warmup + cosine.
14. **Training loop** — two-phase (freeze → unfreeze at epoch 20 with differential LR), **no mixup,
    no SWA**; early stopping on 5-class val macro-F1; restores best weights.
15. **Curves** — loss, macro-F1, per-class F1, LR.
16. **Threshold tuning** — per-class grid search on the clean val set within `[0.30, 0.70]`.
17. **Final test eval** — predict toxic, **derive neutral**, report macro/micro/weighted/samples
    F1, Hamming, ROC-AUC, PR-AUC, and a full per-class classification report (all 5 classes).
18. *(markdown)* ensemble strategy.
19. **Ensemble impl** — `train_with_seed` (consistent with the corrected pipeline) + probability averaging.
20. **Save** — checkpoint (`.pt`) + `v5_summary.json`.
21. *(markdown)* notes on honest evaluation and where future gains live.

---

## 4. Model architecture (~9.74M parameters)

```
word_ids ─► Embedding(VOCAB, 300, frozen←FastText) ─► Linear→256 ─┐
                                                                   ├─ concat ─► SpatialDropout
char_ids ─► CharCNN(emb 32, filters 64 × kernels {2,3,4} = 192) ──┘            │
                                                                                ▼
                                            TextCNN(in=448, filters 224 × {2,3,4})
                                                                                │
                                                                                ▼
                                              BiGRU(hidden 368 × 2 layers, bidir)
                                                                                │
                          ┌──────────── attention ctx ┐  ┌ max-pool ┐  ┌ mean-pool ┐
                          └──────────────────── concat (3 × 736 = 2208) ───────────┘
                                                                                │
                                          Linear→320 ─► (multi-sample dropout) ─► Linear→4
```

**Verified parameter breakdown (VOCAB=9155):**

| Module | Params |
|---|---|
| BiGRU | ~5.10M |
| word_embed (frozen FastText) | ~2.75M |
| TextCNN | ~0.90M |
| fc1 | ~0.74M |
| attention | ~0.59M |
| projection / char_cnn / fc2 | ~0.10M |
| **Total** | **9,742,632 (9.74M)** |

Trainable while embeddings are frozen: ~7.0M; all 9.74M become trainable after the unfreeze epoch.
Counts scale slightly with the realized vocab size; the notebook prints the exact number and an
`OK / ADJUST` flag (target band 8.5M–11M).

---

## 5. Key hyperparameters

| Group | Setting |
|---|---|
| Split | 70 / 15 / 15, multilabel-stratified, seed 42 |
| Max length | 80 tokens; 16 chars/word |
| Embeddings | FastText `cc.bn.300`, frozen until epoch 20, then unfrozen at 0.1× LR |
| Optimizer | AdamW, LR 1.2e-3, weight decay 1e-4 |
| Schedule | 8% warmup + cosine decay; phase-2 linear decay |
| Loss | Focal (γ=2.0), label smoothing 0.03, **no pos_weight** |
| Regularization | spatial dropout 0.35, dropout 0.5, word-dropout 0.15, 5-sample dropout head |
| Augmentation | threat ×2, **train split only** |
| Epochs / early stop | 30 epochs, patience 6 (restores best) |
| Thresholds | per-class grid search in [0.30, 0.70] |

---

## 6. Running on Kaggle (2x T4)

1. **Accelerator:** Notebook → Settings → **GPU T4 x2**. The notebook auto-detects both GPUs and
   wraps the model in `nn.DataParallel`; the effective batch size becomes `64 × num_gpus = 128`.
2. **Dataset:** attach the CSV dataset. The loader checks several common Kaggle paths plus the
   local filename; adjust `Config.DATA_PATH` if your dataset slug differs.
3. **FastText:** cell 10 downloads `cc.bn.300.vec.gz` (~1.2 GB) if it is not already attached as a
   dataset. To skip the download, attach it as a Kaggle dataset and point one of the candidate
   paths at it.
4. **Run all.** Outputs produced: `v5_training_curves.png`, `bengali_cyberbullying_v5_best.pt`,
   `v5_summary.json`.
5. **Optional ensemble:** run `all_probs = [train_with_seed(s) for s in cfg.ENSEMBLE_SEEDS]` then
   `ensemble(all_probs, tuned)` for a small additional gain (needs the GPU budget for 3 runs).

> **Note on multi-GPU:** `nn.DataParallel` is used (single-process, splits each batch across both
> T4s) because it requires no launcher changes and works out-of-the-box in a Kaggle notebook.
> `DistributedDataParallel` is faster but needs a multi-process launch that Kaggle notebooks do
> not provide conveniently.

---

## 7. How this was verified (offline, no GPU)

A CPU harness compiled and executed the notebook's cells:

- **All 17 code cells compile.**
- **Full-data pipeline:** `train∩val = 0`, `train∩test = 0`; derived neutral consistent in every
  split; **total params = 9,742,632 (9.74M)** → within the ~10M band.
- **Full flow on a subsample** (train → threshold tuning → test → save) executes end-to-end;
  predicted `neutral` has **zero** contradictions with toxic predictions; checkpoint and summary
  are written.

(The subsample run uses 2 epochs and zeroed embeddings purely to exercise control flow, so its
metric values are not meaningful — only the real Kaggle T4x2 run produces the reported scores.)

---

## 8. Where further accuracy gains live

- **`troll` / `insult`** are the hardest, most subjective, and mutually overlapping classes —
  the best next step is label cleaning / re-adjudication and targeted error analysis.
- **Backbone upgrade:** if the ~10M-parameter budget can be relaxed, fine-tuning a pretrained
  Bengali transformer (BanglaBERT / MuRIL / XLM-R) is the single biggest lever and also removes
  the ~15% OOV issue inherent to word-level FastText.
- **Manifold mixup** (mixing GRU/feature representations, not token IDs) could be added correctly,
  but needs care under `nn.DataParallel`; it was intentionally left out to keep the pipeline correct.
