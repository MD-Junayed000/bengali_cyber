# Bengali Multi-Label Cyberbullying Detection — v5 (T4x2 Parallel)

## Lightweight Pretrained-Embedding Model (~8.2M total / ~0.75M trainable Phase 1, UNDER 10M)

**5-label multi-label classification:** `vulgar`, `threat`, `troll`, `insult`, `neutral`

---

## Why v5 Exists

v4 achieved **macro-F1 = 0.7323** with a +0.058 train-val gap on ~10.6K samples.
v5 leverages:
1. **New 15K balanced dataset** (+43% more training data)
2. **T4x2 parallel GPU training** (DataParallel → larger effective batch, faster epochs)
3. **Focal Loss** for the severely imbalanced `threat` class (6.36:1 neg/pos)
4. **Threat-class augmentation** (word swap, deletion, char noise → 2x expansion)
5. **Embedding fine-tuning** in the final phase (unfreeze last 30% of training)
6. **Multi-seed ensemble** (3 seeds: 42, 7, 2024) + cross-version (v4+v5) averaging
7. **Cosine Annealing with Warm Restarts + SWA** for better convergence

| Version | Macro-F1 | Train-val gap | Dataset | Key Change |
|---------|----------|---------------|---------|------------|
| v3 | 0.6866 | +0.166 | 10.6K | No semantic priors |
| v4 | 0.7323 | +0.058 | 10.6K | Frozen FastText |
| **v5** | **target ≥0.78** | target <0.06 | **15.2K + augmented** | +data, focal, unfreeze, augment, ensemble, T4x2 |

---

## Parameter Budget (STRICT: under 10M)

| Component | Params | Trainable? |
|-----------|--------|------------|
| FastText embedding 25K × 300 | 7.5M | Phase 1: No / Phase 2: Yes (last 10 epochs) |
| Projection 300 → 128 | 38K | Yes |
| Character CNN | 18K | Yes |
| Word CNN (3 kernels × 96 filters) | 194K | Yes |
| 2-layer BiGRU (hidden=96) | 389K | Yes |
| Attention + classifier | 56K | Yes |
| **Total** | **~8.2M** | **Phase 1: ~0.69M / Phase 2: ~8.2M** |

✅ **UNDER 10M TOTAL PARAMETERS** at all times.

---

## Training & Validation Curves (KEY DELIVERABLE)

The notebook produces a **2×2 subplot figure (14×10 inches)** showing:

1. **[0,0] Training & Validation Loss** — with unfreeze epoch vertical line
2. **[0,1] Training & Validation Macro-F1** — with best epoch marker + unfreeze line
3. **[1,0] Per-Class Validation F1** — 5 colored lines for vulgar/threat/troll/insult/neutral
4. **[1,1] Learning Rate Schedule** — log-scale, showing warmup → cosine → SWA phases

Plus overfitting diagnostic comparing train-val gap to v4 baseline.

---

## T4x2 Parallel GPU Training

### How DataParallel Works on 2× T4:
1. Model is replicated on both GPUs automatically
2. Each mini-batch (128) is split: 64 samples → GPU0, 64 samples → GPU1
3. Forward pass runs simultaneously on both GPUs
4. Gradients are gathered on GPU0, averaged, and model updated
5. ~1.7-1.8× speedup (not perfect 2× due to communication overhead)

### Practical Implementation:
```python
# After model creation
model = V5Model(cfg, embed_matrix, VOCAB_SIZE, CHAR_VOCAB_SIZE)
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Effective batch = 64 per GPU × 2 GPUs = 128
EFFECTIVE_BATCH = cfg.BATCH_SIZE_PER_GPU * NUM_GPUS  # 128

# DataLoader with drop_last=True for consistent batch sizes
train_loader = DataLoader(train_ds, batch_size=EFFECTIVE_BATCH,
                          shuffle=True, num_workers=4, pin_memory=True,
                          drop_last=True)
```

### Timing Expectations:
- v4 on T4×1: ~4.4s per epoch (7,404 train samples)
- v5 on T4×2: ~4-5s per epoch (~11,000 train samples with augmentation)
- Total training: 35 epochs × 5s ≈ ~3 minutes

---

## Threat-Class Augmentation Strategy

### Problem:
- `threat` class has only 2,062 samples (13.58%), creating 6.36:1 neg/pos ratio
- In v4, threat was one of the weaker performers

### Solution: Augment threat samples by 2× using:
1. **Random word swap** — swap positions of 2 random words in the sentence
2. **Random word deletion** — remove individual words with probability 0.1
3. **Synonym-style perturbation** — duplicate a random Bengali word with slight character-level noise (adjacent char swap)

### Implementation:
```python
def augment_threat_samples(df, cfg):
    threat_df = df[df['threat'] == 1].copy()
    n_augment = int(len(threat_df) * (cfg.THREAT_AUG_FACTOR - 1))  # ~2062 new samples

    aug_rows = []
    for _ in range(n_augment):
        row = threat_df.sample(1).iloc[0].copy()
        row[cfg.TEXT_COL] = augment_text(str(row[cfg.TEXT_COL]))
        aug_rows.append(row)

    return pd.concat([df, pd.DataFrame(aug_rows)], ignore_index=True)
```

### Expected Impact:
- Threat samples: 2,062 → ~4,124 (now 24% of dataset instead of 13.6%)
- Threat neg/pos ratio: 6.36:1 → ~3.2:1 (much more manageable)
- Combined with focal loss, threat F1 should improve from 0.77 → 0.80+

---

## Ensemble Strategy

### 1. Multi-Seed Ensemble (Primary)
Train 3 models with seeds `[42, 7, 2024]` and average their predicted probabilities:
```python
all_probs = [train_with_seed(s) for s in [42, 7, 2024]]
avg_probs = np.mean(all_probs, axis=0)
preds = (avg_probs >= thresholds).astype(int)
```
**Expected improvement:** +0.5-1.5% macro-F1

### 2. Cross-Version Ensemble (v4 + v5)
Average probability outputs from the best v4 model and v5 model:
```python
cross_probs = 0.5 * v4_test_probs + 0.5 * v5_test_probs
preds = (cross_probs >= thresholds).astype(int)
```
**Expected improvement:** +0.5-1.0% macro-F1 (captures complementary patterns)

### Why It Works:
- Different seeds → different local minima → different error patterns
- v4 (frozen) and v5 (unfrozen) learn different feature spaces
- Averaging smooths prediction noise → lower variance → better generalization

---

## Complete Notebook Sections (20 Sections)

### Section 1 — Title & Introduction (Markdown)
- Bengali Multi-Label Cyberbullying Detection v5
- Architecture overview, hardware, dataset, key features

### Section 2 — Setup & Imports (Code)
- `!pip install iterative-stratification -q`
- All imports: torch, numpy, pandas, sklearn, matplotlib, seaborn
- `set_seed()` function for reproducibility
- Device detection: print GPU count and names

### Section 3 — Configuration (Code)
```python
class Config:
    DATA_PATH = 'combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv'
    LABEL_COLS = ['vulgar', 'threat', 'troll', 'insult', 'neutral']
    NUM_CLASSES = 5
    VOCAB_SIZE = 25000        # Keeps total params at ~8.2M (<10M)
    MIN_FREQ = 2
    MAX_LEN = 80
    FASTTEXT_DIM = 300
    FREEZE_EMBEDDING = True
    UNFREEZE_AT_EPOCH = 25
    UNFREEZE_LR_FACTOR = 0.1  # 10x smaller LR for embeddings when unfrozen
    PROJECTION_DIM = 128
    CNN_FILTERS = 96
    CNN_KERNELS = (2, 3, 4)
    GRU_HIDDEN = 96
    GRU_LAYERS = 2
    DROPOUT_EMB = 0.35
    DROPOUT = 0.5
    NUM_DROPOUT_SAMPLES = 5
    BATCH_SIZE_PER_GPU = 64   # 64 × 2 GPUs = 128 effective
    EPOCHS = 35
    LR = 1.5e-3
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    AUGMENT_THREAT = True
    THREAT_AUG_FACTOR = 2.0
    USE_SWA = True
    PATIENCE = 8
    ENSEMBLE_SEEDS = [42, 7, 2024]
```

### Section 4 — Load & Clean Dataset (Code)
- Load 15K CSV
- Fix 6 neutral+toxic contradictions
- Merge 64 duplicate texts
- Drop 98 single-word samples
- Print per-class counts

### Section 5 — Threat-Class Augmentation (Markdown + Code)
- `random_swap()`, `random_deletion()`, `char_noise()`, `synonym_perturbation()`
- `augment_threat_samples(df, cfg)` — adds ~2,062 augmented threat rows
- Print before/after class distribution

### Section 6 — EDA (Code)
- 3-panel figure: per-class counts, labels/comment, word count distribution
- Verify MAX_LEN coverage

### Section 7 — Text Preprocessing (Code)
- Bengali regex, URL/mention/emoji/punct removal
- `clean_text()` and `tokenize()` functions
- Apply to dataframe, report token stats

### Section 8 — Stratified Split 70/10/20 (Code)
- `MultilabelStratifiedShuffleSplit` for proper multi-label stratification
- Print sizes and per-class rates across splits
- Expected: ~11,200 train / ~1,600 val / ~3,200 test (after augmentation)

### Section 9 — Vocabularies (Code)
- `build_word_vocab()` — min_freq=2, cap=25K (training only, no leakage)
- `build_char_vocab()` — 250 characters
- `encode_words()`, `encode_chars()` functions
- Report OOV rate on validation set

### Section 10 — Load FastText Embeddings (Code)
- Find or download `cc.bn.300.vec.gz` (~840MB)
- Stream-parse: only keep vectors for training vocab words
- Build embedding matrix, report coverage %
- Expected: >88% coverage with larger vocab

### Section 11 — Dataset & DataLoaders (Code)
- `BengaliCBDataset` class with word dropout
- T4×2 optimized: batch=128, num_workers=4, pin_memory=True, drop_last=True
- Print batch counts

### Section 12 — Model Architecture (Code)
- `SpatialDropout1D` — drop entire embedding channels
- `CharCNN` — 3 kernel sizes (2,3,4) × 32 filters
- `AdditiveAttention` — Bahdanau-style attention
- `V5Model` — full architecture with multi-sample dropout classifier
- **Verify: total params < 10M** (prints PASS/FAIL)
- Wrap with `nn.DataParallel` for T4×2

### Section 13 — Focal Loss & Training Setup (Code)
- `FocalBCELoss(gamma=2.0, pos_weight, smoothing=0.05)`
- Compute per-class pos_weight from training labels
- AdamW optimizer (only trainable params in Phase 1)
- LambdaLR scheduler: warmup (8%) → cosine → SWA
- SWA via `torch.optim.swa_utils.AveragedModel`

### Section 14 — Training Loop (Code) — MAIN LOOP
- **Phase 1 (epochs 1-24):** Frozen embeddings, normal LR
- **Phase 2 (epochs 25-35):** Unfreeze embeddings, differential LR (10× smaller for embeddings)
- Per-epoch: mixup augmentation, multi-sample dropout, gradient clipping
- SWA model updates when `global_step >= swa_start_step`
- Early stopping on val F1 (patience=8)
- Track: train_loss, val_loss, train_f1, val_f1, val_per_class_f1, lr, phase
- Print per-epoch: `Epoch XX [Phase] | TrLoss | VaLoss | TrF1 | VaF1 | Gap | LR | Time`

### Section 15 — Training & Validation Curves (Code) — KEY DELIVERABLE
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
# [0,0]: Train/Val Loss with unfreeze line
# [0,1]: Train/Val Macro-F1 with best epoch + unfreeze lines
# [1,0]: Per-class Val F1 (5 colored lines)
# [1,1]: Learning Rate (log scale)
plt.savefig('v5_training_curves.png', dpi=150)
```
Plus overfitting diagnostic.

### Section 16 — Threshold Tuning (Code)
- Grid search: `np.arange(0.2, 0.8, 0.02)` per class
- Print tuned thresholds and improvement over default 0.5

### Section 17 — Final Test Evaluation (Code)
- `evaluate_with_thresholds()` on test set
- Print: Macro-F1, Micro-F1, Weighted-F1, Samples-F1, Hamming Loss, ROC-AUC, PR-AUC
- Full `classification_report` per class

### Section 18 — Ensemble Strategy (Markdown + Code)
- `train_with_seed(seed)` function — trains full model, returns test probabilities
- `ensemble_predictions(all_probs, thresholds)` — averages probs, applies thresholds
- Instructions for multi-seed and cross-version ensemble
- Note: single-seed for quick runs, full ensemble for best accuracy

### Section 19 — Save Model & Summary (Code)
- Save `.pt` checkpoint (state_dict, config, thresholds, history, vocabs)
- Save `v5_summary.json` with all metrics
- Print final macro-F1 and param count confirmation

### Section 20 — Conclusion (Markdown)
- Version comparison table (v3 vs v4 vs v5)
- Key improvements listed
- Thesis recommendations
- Future work suggestions

---

## Key Differences: v4 → v5

| Aspect | v4 | v5 |
|--------|----|----|
| Dataset | 10,594 samples | **~17,000 samples (with augmentation)** |
| GPUs | 1× T4 | **2× T4 (DataParallel)** |
| Batch size | 64 | **128 (64 per GPU)** |
| Loss | Smoothed BCE | **Focal Loss (γ=2)** |
| Embedding | Frozen forever | **Unfreeze at epoch 25** |
| Threat handling | pos_weight only | **pos_weight + Focal + Augmentation (2×)** |
| Vocab min_freq | 1 | **2** |
| Epochs | 30 | **35** |
| Dropout | 0.6 | **0.5** (more data = less overfitting) |
| LR | 1e-3 | **1.5e-3** (larger batch) |
| Scheduler | Warmup+Cosine+SWA | **Warmup+Cosine+SWA** |
| Patience | 6 | **8** |
| Ensemble | None | **3-seed + cross-version** |
| Total params | 8.2M (<10M ✓) | **8.2M (<10M ✓)** |

---

## Expected Performance

| Metric | v4 (actual) | v5 Single-Seed (expected) | v5 Ensemble (expected) |
|--------|-------------|---------------------------|------------------------|
| Macro-F1 | 0.7323 | 0.76-0.78 | 0.78-0.80 |
| Train-Val Gap | +0.058 | <0.06 | — |
| Threat F1 | 0.7719 | 0.80+ | 0.82+ |
| Troll F1 | 0.6045 | 0.64-0.68 | 0.66-0.70 |
| Vulgar F1 | 0.8338 | 0.85+ | 0.86+ |

---

## How to Run on Kaggle

1. Create new Kaggle notebook
2. Upload `bengali-cyberbullying-v5-t4x2.ipynb`
3. Add the 15K balanced CSV dataset as input
4. Add FastText Bengali vectors dataset OR enable internet (for download)
5. **Select accelerator: GPU T4 × 2**
6. Run All → ~3-5 minutes total training
7. Outputs: training curves PNG, checkpoint .pt, summary .json
