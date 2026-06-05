# Bengali Multi-Label Cyberbullying Detection — Implementation Plan

Two complementary notebooks are provided. Both share the same correct, leakage-free data pipeline,
predict the **4 toxic labels** (`vulgar`, `threat`, `troll`, `insult`) and **derive** `neutral` as
`NOT(any toxic)`, tune per-class thresholds, and run on **Kaggle T4 x2**.

| Notebook | Backbone | Params | Honest Macro-F1 (expected) | Use when |
|---|---|---|---|---|
| `bengali-cyberbullying-lightweight-v4.ipynb` | CharCNN+FastText+TextCNN+BiGRU+Attention (from scratch) | **~9.5M** | ~0.74–0.77 | the ~10M budget is required |
| `bengali-cyberbullying-transformer.ipynb` | Pretrained BanglaBERT / MuRIL / XLM-R | ~110M | ~0.82–0.88 | accuracy matters more than size |

> **Honest-baseline context.** After the data-leakage fix, the lightweight model's *true* score was
> Macro-F1 ≈ **0.72** (the earlier 0.78 was inflated by augmented `threat` near-duplicates leaking
> into the test set — its test support dropped 646 → 321 once fixed). Neither notebook leaks; the
> numbers they report are trustworthy.

---

## 0. Data & labels

- Dataset: `combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral_plus_threat300.csv`
  (~15.5k rows → ~15.3k after cleaning + dedup on cleaned text).
- 4 toxic labels are modeled; `neutral` is derived, so `neutral`+toxic contradictions are impossible.
- Class rates (full): vulgar ~24.9%, threat ~14.0%, troll ~28.4%, insult ~28.5%, neutral ~35.7%.

Both notebooks **split before doing anything else** and assert `train ∩ val = train ∩ test = 0`.

---

## 1. Track 1 — lightweight ~10M (improved, `bengali-cyberbullying-lightweight-v4.ipynb`)

### Why it changed
The previous corrected run was leakage-free but **overfit early**: validation Macro-F1 peaked at
epoch ~11 (≈0.72) while training F1 ran to 0.86. A ~10M-param model is over-capacity for only ~12k
short comments. This version attacks overfitting directly.

### What changed (vs the previous corrected run)
1. **All-class train-only augmentation** (was threat-only). Every category is augmented up to a
   per-class target (rarer classes pulled up more) via word swap / deletion / pseudo-synonym, which
   expands the training set from ~12k to **~20.7k** so the 10M-param model has enough data.
2. **Vocab is built on the *original* (pre-augmentation) training text.** This is important: the
   char-noise augmentation invents tokens; if the vocab were built after augmentation it grew to
   ~14k and pushed the model to **11.3M params**. Building it on the original text keeps the
   embedding bounded and the total at **~9.45M**; augmented-only tokens map to `<UNK>`.
3. **Stronger regularization:** word-dropout 0.15→0.25, embedding spatial-dropout 0.35→0.40,
   weight-decay 1e-4→3e-4.
4. **Calmer optimization:** peak LR 1.2e-3→7e-4, warmup 8%→12% (smooths the noisy val curve).
5. **Earlier embedding unfreeze** (epoch 20→7 at 0.1× LR) so phase-2 actually runs before early stop.

### Architecture (~9.45M params; ~10M target met)
```
word_ids ─► Embedding(VOCAB≈8–9k, 300, frozen←FastText) ─► Linear→256 ─┐
                                                                        ├─ concat ─► SpatialDropout
char_ids ─► CharCNN(emb 32, filters 64 × {2,3,4} = 192) ───────────────┘            │
                                              TextCNN(filters 224 × {2,3,4}) ────────┘
                                              BiGRU(hidden 368 × 2, bidir)
                       attention ctx ⊕ max-pool ⊕ mean-pool ─► Linear→320 ─► (5× dropout) ─► Linear→4
```
Embedding (frozen FastText) is the only piece that scales with vocab; everything else is fixed, so
the parameter count stays in the 8.5–11M band (the notebook prints the exact count + an OK/ADJUST flag).

### Loss & decoding
Focal loss (γ=2.0, smoothing 0.05) **without** `pos_weight` (balance comes from augmentation), then
per-class threshold tuning in `[0.30, 0.70]`. Multi-seed ensemble (`train_with_seed`) is provided.

---

## 2. Track 2 — transformer (`bengali-cyberbullying-transformer.ipynb`)

The real accuracy unlock: fine-tune a pretrained Bengali encoder.

- **Backbone:** tries `csebuetnlp/banglabert` → `google/muril-base-cased` → `xlm-roberta-base`
  (first that loads wins). ~110M params — **intentionally over the 10M budget**.
- **Head:** encoder → masked **mean-pooling** → dropout → `Linear(hidden, 4)`. Loss = multi-label
  Focal (γ=2.0). Neutral derived exactly as in Track 1.
- **Cleaning is lighter** than the lightweight model (keeps punctuation/natural text; only strips
  URLs/mentions/emoji) because transformers benefit from natural text.
- **Optimization:** AdamW with no weight-decay on bias/LayerNorm, **higher LR for the new head**
  (1e-4) vs encoder (2e-5), linear warmup schedule, **mixed precision (AMP)**, gradient clipping,
  4 epochs with early stopping. **No synthetic augmentation** (pretrained + few epochs + weight
  decay generalizes well on ~12k samples).
- **Multi-GPU:** `nn.DataParallel` over both T4s; AMP keeps memory in budget. Lower
  `BATCH_SIZE_PER_GPU` or `MAX_LEN` if you hit OOM.
- For best BanglaBERT results, optionally add the authors' normalizer
  (`pip install git+https://github.com/csebuetnlp/normalizer`) inside `light_clean`.

---

## 3. Running on Kaggle (both notebooks)

1. Accelerator: **GPU T4 x2**. Both notebooks auto-detect 2 GPUs and wrap the model in
   `nn.DataParallel` (effective batch = per-GPU batch × 2).
2. Attach the CSV dataset; the loaders probe several common Kaggle paths plus the local filename.
3. Lightweight notebook downloads FastText `cc.bn.300.vec.gz` (~1.2 GB) on first run (attach it as a
   dataset to skip). Transformer notebook downloads the HF model weights (~110M) on first run.
4. Run All. Outputs:
   - Lightweight: `v6_training_curves.png`, `bengali_cyberbullying_v6_lightweight.pt`, `v6_summary.json`.
   - Transformer: `transformer_curves.png`, `bengali_transformer_best.pt`, `transformer_summary.json`.
5. Optional ensemble (lightweight): `all_probs = [train_with_seed(s) for s in cfg.ENSEMBLE_SEEDS]`.

---

## 4. Offline verification performed (CPU, no GPU / no big downloads)

Both notebooks were executed cell-by-cell on CPU via a harness (FastText stubbed for Track 1; a tiny
`prajjwal1/bert-tiny` backbone substituted for Track 2; training subsampled to a couple hundred rows
and 1–2 epochs purely to exercise control flow):

- **Track 1:** all 17 code cells compile; on **full data** `train∩val=0`, `train∩test=0`, neutral
  consistent, training expands 10,677 → **20,707**, and **total params = 9,450,732 (9.45M, "OK")**.
  The full train→threshold→test→save flow runs and predicted `neutral` has **0 contradictions**.
- **Track 2:** all 12 code cells compile; tokenizer → model → AMP-aware train loop → threshold →
  test → save runs end-to-end; predicted `neutral` has **0 contradictions**; artifacts saved.

(Subsampled/stub metric values are meaningless by design — only the real Kaggle T4x2 runs produce the
reported scores.)

---

## 5. Where remaining accuracy lives

- `troll` and `insult` are the hardest, most subjective, and mutually overlapping classes (~1000
  co-occurrences). This is a **label-quality ceiling** for both tracks — error analysis and
  re-adjudication of ambiguous `troll`/`insult` cases is the highest-value data-side step.
- Track 2 (transformer) is the biggest model-side lever and also removes the ~15% OOV problem that
  word-level FastText cannot solve.
- Optional: ensemble the two tracks (average probabilities) for a small additional, complementary gain.
