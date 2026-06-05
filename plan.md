# Bengali Multi-Label Cyberbullying Detection — Implementation Plan

Two notebooks share one correct, leakage-free pipeline. Both predict the **4 toxic labels**
(`vulgar`, `threat`, `troll`, `insult`) and **derive** `neutral` as `NOT(any toxic)`, tune per-class
thresholds, and run on **Kaggle T4 x2**.

| Notebook | Backbone | Params | Result so far | Notes |
|---|---|---|---|---|
| `bengali-cyberbullying-lightweight-v4.ipynb` (v8) | CharCNN+FastText+TextCNN+BiGRU+Attn | **10.03M** | v7 was 0.713; v8 fixes the issues + ensemble | from-scratch, data-limited |
| `bengali-cyberbullying-transformer.ipynb` | BanglaBERT / MuRIL / XLM-R | ~110M | 0.7654 (undertrained) → ~0.78–0.82 expected | the accuracy track |

---

## Track 1 (v8.1) — corrections to the "still worst" lightweight model

Across runs the from-scratch model plateaued ~0.71–0.72 and overfit after its best epoch (v8 peaked
at val 0.7236 @ epoch 10, then val oscillated 0.70–0.72 while train F1 climbed to 0.80). Fixes
(all verified offline):
1. **Balance-preserving augmentation** — every training row gets the same number of augmented copies,
   so the class distribution is preserved. Verified: post-aug train neutral rate = **0.358** (matches
   the data; the earlier per-class-target aug had skewed it to 0.22, hurting `neutral`).
2. **Lower post-unfreeze LR** — body 0.30×, embeddings 0.05× on a cosine decay (regularizes the
   unfreeze phase that previously caused runaway overfitting).
3. **SWA (Stochastic Weight Averaging)** — averages weights over the post-unfreeze plateau and is
   kept **only if it beats best-single on validation**, so it cannot hurt. Directly targets the
   "noisy plateau / single spiky peak" symptom and usually transfers to test better than one epoch.
4. **Right-sized + heavily regularized** — projection 256, char-CNN 64, TextCNN 224, BiGRU 384×2,
   FC 384; dropout 0.5, spatial-dropout 0.45, word-dropout 0.30, weight-decay 1e-3.
   **Total parameters = 10,031,084 (10.03M)**, printed in Section 12.
5. **Multi-seed ensemble runs by default** (seeds 42/7/2024, each SWA-selected; seed 42 reuses the
   already-trained main model): averages toxic probabilities, re-tunes thresholds on averaged val,
   reports the **ensemble** test score as the headline (Section 19).

> Honest ceiling: subjective, overlapping `troll`/`insult` labels cap this around ~0.73–0.75.
> The transformer is where ~0.80 lives.

## Track 2 — transformer hang fixed + best-practice fine-tuning

**Why it was stuck at "model.safetensors 100% … ====":** the first epoch never started because a
**fast (Rust) tokenizer was called inside `__getitem__` with `DataLoader(num_workers=2)`** — forking
workers around a fast tokenizer deadlocks (the `TOKENIZERS_PARALLELISM` fork issue). It hung on the
first batch, not in training.

Fixes (verified offline):
1. `os.environ['TOKENIZERS_PARALLELISM'] = 'false'`.
2. **Pre-tokenize each split once** (batched) and store tensors; `__getitem__` only indexes tensors —
   no tokenizer call in workers, and faster.
3. `num_workers=0` for all loaders.

High-accuracy recipe (fixes the earlier undertraining at 4 epochs): **8 epochs + early stopping**,
**layer-wise LR decay (LLRD ×0.9)**, **mean+max pooling**, **multi-sample dropout head**,
**FGM adversarial training**, cosine schedule, AMP, `nn.DataParallel`. Backbone fallback
BanglaBERT → MuRIL → XLM-R. ~110M params (intentionally over the 10M budget).

---

## Running on Kaggle (both)
1. Accelerator **GPU T4 x2** + **Internet ON** (weights / FastText download).
2. Attach the CSV dataset (loaders probe common paths + local filename).
3. Run All. Outputs:
   - Lightweight: `v8_training_curves.png`, `bengali_cyberbullying_v8_lightweight.pt`, `v8_summary.json`.
   - Transformer: `transformer_curves.png`, `bengali_transformer_best.pt`, `transformer_summary.json`.
4. Lightweight runtime: the default ensemble trains 3 models (set `RUN_ENSEMBLE=False` for a single run).
5. Transformer OOM? lower `BATCH_SIZE_PER_GPU` (8) / `MAX_LEN` (96), or `USE_FGM=False` (FGM ~2× step time).

## Offline verification performed (CPU, no GPU / no big downloads)
- **Track 1:** 17 cells compile; full data → zero leakage, neutral consistent, **balance preserved
  (neutral 0.358)**, **params 10,031,084**; full train→threshold→test→**ensemble**→save runs with 0
  neutral contradictions.
- **Track 2:** 12 cells compile; dataset confirmed pre-tokenized (`__getitem__` indexes tensors only),
  `num_workers=0`; full tokenize→model(mean+max+multi-sample)→LLRD→FGM→threshold→test→save runs with
  0 neutral contradictions. (Stub-backbone metrics are meaningless by design.)

## Where remaining accuracy lives
- `troll`/`insult` overlap is the data-quality ceiling for both tracks — relabeling/error analysis is
  the top data-side step.
- Track 2 is the biggest model-side lever; for more, try `csebuetnlp/banglabert_large` (smaller batch).
- Ensembling the two tracks' probabilities can add a small complementary gain.
