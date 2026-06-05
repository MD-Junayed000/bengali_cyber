# Bengali Multi-Label Cyberbullying Detection — Implementation Plan

Two notebooks share the same correct, leakage-free pipeline. Both predict the **4 toxic labels**
(`vulgar`, `threat`, `troll`, `insult`) and **derive** `neutral` as `NOT(any toxic)`, tune per-class
thresholds, and run on **Kaggle T4 x2**.

| Notebook | Backbone | Params | Last measured Test Macro-F1 | Use when |
|---|---|---|---|---|
| `bengali-cyberbullying-lightweight-v4.ipynb` | CharCNN+FastText+TextCNN+BiGRU+Attention | **15.24M** (cap removed) | 0.714 (prev 9.45M run) | a compact, from-scratch model is wanted |
| `bengali-cyberbullying-transformer.ipynb` | BanglaBERT / MuRIL / XLM-R + best-practice FT | ~110M | 0.765 (undertrained 4 ep) → ~0.78–0.82 expected | accuracy is the priority |

> **Measured results so far (honest, leakage-free):** lightweight 9.45M → **0.7143**; BanglaBERT
> 110M (4 epochs, still improving) → **0.7654**. The transformer clearly wins and was undertrained,
> which is exactly what the upgraded version fixes.

---

## 0. Data & labels
- `combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral_plus_threat300.csv`
  (~15.5k → ~15.3k after cleaning + dedup on cleaned text).
- 4 toxic labels modeled; `neutral` derived (no neutral+toxic contradictions).
- Both notebooks **split before anything** and assert `train ∩ val = train ∩ test = 0`.

---

## 1. Task 1 — lightweight, **10M cap removed** (`bengali-cyberbullying-lightweight-v4.ipynb`)

### What changed in this version
- **Removed the ~10M parameter cap** and scaled the architecture up:
  projection 256→**300**, char-CNN 64→**96** filters (embed 32→48), TextCNN 224→**256**,
  BiGRU 368→**512** hidden (2 layers), FC 320→**512**.
- **Exact parameter count utilized = 15,235,016 (15.24M).** Breakdown (verified on full data):
  - Word embedding (FastText, 8,182 × 300): **2,454,600** (frozen until epoch 6, then trainable)
  - Rest of the network: **12,780,416**
  - The notebook prints this table in Section 12 and repeats the total in the final summary.
- **Fixed the learning-rate collapse at unfreeze.** Previously the body LR dropped ~3× and decayed
  linearly after unfreezing, so it barely trained. Now the body keeps its full LR on a single cosine
  decay over the remaining steps; only embeddings use a 0.1× LR.
- **Stronger regularization** so the bigger model doesn't overfit ~12k comments: dropout 0.5,
  embedding spatial-dropout 0.45, word-dropout 0.30, weight-decay 1e-3.

### Kept from before
Split-before-augment with hard zero-leakage assertions; all-class train-only augmentation
(~12k → ~20.7k, vocab built on the **original** text so augmentation noise can't inflate it);
Focal loss without `pos_weight`; derived neutral; per-class threshold tuning; multi-seed ensemble.

### Honest expectation
This is a from-scratch model on a small, partly-subjective dataset. Going 9.45M → 15.24M mainly
changes the **reported parameter count**, not the score (expect ~0.71–0.75; the model is
data/label-limited, with `troll`/`insult` overlap as the ceiling). For best lightweight numbers,
run the multi-seed ensemble (Section 19). For a real accuracy gain, use Task 2.

---

## 2. Task 2 — transformer, **upgraded to best** (`bengali-cyberbullying-transformer.ipynb`)

The previous run hit 0.7654 but was **cut off at 4 epochs while still improving**. The upgrade adds
the standard high-accuracy fine-tuning recipe:

1. **8 epochs + early stopping (patience 3)** — trains to convergence.
2. **Layer-wise LR decay (LLRD, ×0.9/layer)** — top layers full LR, lower layers decayed, head at
   1e-4. Most reliable single transformer-FT trick. (Builds one param group per layer + embeddings + head.)
3. **Mean + max pooling** (concatenated) instead of mean only.
4. **Multi-sample dropout head** (5×).
5. **FGM adversarial training** — perturbs word embeddings each step (toggle `USE_FGM`); ~2× step
   time, consistently lifts text-classification F1.
6. **Cosine schedule + warmup**, AMP, `nn.DataParallel` over both T4s.

Backbone: `csebuetnlp/banglabert` → `google/muril-base-cased` → `xlm-roberta-base` (first that
loads). ~110M params — **intentionally over the 10M budget**. Light cleaning only (keeps natural
text). No synthetic augmentation. **Expected ~0.78–0.82** (vs 0.7654 undertrained).

---

## 3. Running on Kaggle (both)
1. Accelerator **GPU T4 x2** (auto-detected → `nn.DataParallel`).
2. Attach the CSV dataset; loaders probe common Kaggle paths + local filename.
3. Lightweight downloads FastText `cc.bn.300.vec.gz` (~1.2 GB) on first run; transformer downloads
   the HF weights (~110M). Enable Internet in notebook settings, or attach them as datasets.
4. Run All. Outputs:
   - Lightweight: `v7_training_curves.png`, `bengali_cyberbullying_v7_lightweight.pt`, `v7_summary.json`.
   - Transformer: `transformer_curves.png`, `bengali_transformer_best.pt`, `transformer_summary.json`.
5. Transformer OOM on T4? lower `BATCH_SIZE_PER_GPU` (8) or `MAX_LEN` (96), or set `USE_FGM=False`.

---

## 4. Offline verification performed (CPU, no GPU / no big downloads)
- **Task 1:** 17 cells compile; full data → `train∩val=0`, `train∩test=0`, neutral consistent,
  training expands 10,677 → 20,707, **total params = 15,235,016 (15.24M)**; full
  train→threshold→test→save runs with **0** neutral contradictions.
- **Task 2:** 12 cells compile; with a tiny stub backbone the full
  tokenize→model(mean+max+multi-sample)→**LLRD optimizer**→**FGM training**→threshold→test→save
  runs with **0** neutral contradictions. (Stub metrics are meaningless by design.)

---

## 5. Where remaining accuracy lives
- `troll`/`insult` are the hardest, most subjective, overlapping classes (~1000 co-occurrences) — a
  label-quality ceiling for both tracks; error analysis / re-adjudication is the top data-side step.
- Task 2 is the biggest model-side lever (and removes the ~15% OOV of word-level FastText). For an
  even higher ceiling try `csebuetnlp/banglabert_large` with a smaller batch / MAX_LEN.
- Ensembling the two tracks' probabilities can add a small complementary gain.
