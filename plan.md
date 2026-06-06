# Bengali Multi-Label Cyberbullying Detection — Implementation Plan

Two notebooks share one correct, leakage-free pipeline. Both run on **Kaggle T4 x2**, **derive**
`neutral` as `NOT(any toxic)`, tune per-class thresholds, and (NEW) default to the **troll+insult →
`flaming` merge**. This plan reflects the notebooks **as you last uploaded them** plus the merge.

## NEW: troll + insult → `flaming` (default `MERGE_TROLL_INSULT=True`)

`troll` and `insult` were the weakest, most-confused classes in every run. The data shows why:
they **overlap ~23%** (troll 4350, insult 4449, both 1005) and are subjective. Merging them into a
single **`flaming`** class (a recognized cyberbullying category = hostile/insulting provocation):
- `flaming = troll OR insult` = **7794 rows (50.2%)** — a large, clean class.
- New label set: **`vulgar`, `threat`, `flaming`** (3 toxic) + derived `neutral` (4 classes total).
- Removes the single hardest decision boundary → macro-F1 should rise materially
  (lightweight ~0.72 → ~0.78; transformer ~0.775 → ~0.83, estimated).

**Honesty note (for the thesis):** part of this gain comes from *simplifying the label space*
(4 classes instead of 5, dropping the confusable pair), not purely from a better model. Set
`cfg.MERGE_TROLL_INSULT = False` (Section 2) to reproduce the original 5-class numbers — **report
both** so the comparison is transparent.

Implementation: applied once in the data cell — `flaming = troll|insult`, then
`cfg.TOXIC_COLS / LABEL_COLS / NUM_OUT` are reassigned dynamically, so every downstream cell (model
head width, neutral derivation, threshold tuning, reports) adapts automatically.

| Notebook | Params | last 5-class result | with `flaming` merge (est.) |
|---|---|---|---|
| `...-lightweight-v4.ipynb` | 10.0M | val 0.7236 / test ~0.71 | ~0.78 |
| `...-transformer.ipynb` | ~110M | val 0.7749 | ~0.83 |

---

## Track 1 (lightweight) — your uploaded v8.1
- Split-before-augment with **zero-leakage assertions**; vocab from original (pre-aug) text.
- **Balance-preserving augmentation** (each row ×1 copy → distribution preserved, neutral 0.358).
- **Lower post-unfreeze LR** (body 0.30×, embeddings 0.05×) to curb overfitting.
- **SWA** over the post-unfreeze plateau, kept only if it beats best-single on validation
  (last run: SWA 0.7219 < best-single 0.7236 → best-single kept, as designed — SWA can't hurt).
- **Multi-seed ensemble** (42/7/2024, each SWA-selected; seed 42 reuses the main model) — headline.
- Focal loss without `pos_weight`; per-class thresholds; `nn.DataParallel` (T4 x2). ~10M params.

## Track 2 (transformer) — your uploaded (lean) version + merge
- Encoder (BanglaBERT → MuRIL → XLM-R fallback) + masked **mean-pool** + linear head, Focal loss,
  AdamW (no WD on bias/LayerNorm, higher head LR), linear warmup schedule, AMP, `nn.DataParallel`.
- 8 epochs + early stopping; this is the version you ran (≈81 s/epoch, best val **0.7749**).
- Added a one-line `TOKENIZERS_PARALLELISM=false` safety guard (you hit a tokenizer/DataLoader stall
  once; this is a harmless mitigation).
- **Optional higher-accuracy variant in git history** (commit `e4b2691`): adds **LLRD**,
  **mean+max pooling**, **multi-sample dropout**, **FGM adversarial training**, cosine schedule, and
  pre-tokenized loaders. It typically adds ~0.01–0.02 macro-F1 but ~doubles epoch time (FGM). Restore
  it with `git checkout e4b2691 -- bengali-cyberbullying-transformer.ipynb` if you want to try it.

---

## Running on Kaggle (both)
1. Accelerator **GPU T4 x2** + **Internet ON** (weights / FastText download).
2. Attach the CSV dataset (DATA_PATH is set to your Kaggle dataset path; loaders also probe fallbacks).
3. Run All. Outputs: `v8_*` (lightweight) and `transformer_*` (transformer) checkpoints/summaries/curves.
4. To compare label setups, run once with `MERGE_TROLL_INSULT=True` (default) and once `False`.

## Offline verification performed (CPU, no GPU / big downloads)
- Both notebooks **compile** and run end-to-end on a subsample (stubbed embeddings / tiny backbone).
- **Merge confirmed**: `cfg.NUM_OUT=3`, classes `[vulgar, threat, flaming, neutral]`, flaming ≈ 0.50.
- **Zero leakage**, predictions are 4 columns, `neutral` derived with **0 contradictions**, the
  Track-1 ensemble path runs.

## Where remaining accuracy lives
- With `flaming` merged, `threat` (rarest, ~14%) becomes the next class to watch.
- The FGM/LLRD transformer variant (history) is the highest from-the-shelf ceiling; or try
  `csebuetnlp/banglabert_large` with a smaller batch / MAX_LEN.
- Ensembling the two tracks' probabilities can add a small complementary gain.
