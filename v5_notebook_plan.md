# Bengali Multi-Label Cyberbullying Detection — v5.1 (T4x2 Parallel, Troll Fix)

## Lightweight Pretrained-Embedding Model (~3.5-4M total, UNDER 10M)

**5-label multi-label classification:** `vulgar`, `threat`, `troll`, `insult`, `neutral`

---

## Version History & Results

| Version | Macro-F1 | Troll F1 | Threat F1 | Train-Val Gap | Dataset | Key Change |
|---------|----------|----------|-----------|---------------|---------|------------|
| v3 | 0.6866 | ~0.55 | ~0.68 | +0.166 | 10.6K | No semantic priors |
| v4 | 0.7323 | 0.6045 | 0.7719 | +0.058 | 10.6K | Frozen FastText |
| v5 | **0.7791** | 0.6187 | **0.8898** | +0.072 | 15.2K+aug | Focal loss, augment, T4x2 |
| **v5.1** | **target 0.82+** | **target 0.72+** | 0.89+ | target <0.05 | 15.2K+aug | **Troll fix: sarcasm features, label interaction** |

---

## The Troll Problem (v5's Worst Issue)

### Diagnosis from v5 Output:
```
              precision    recall  f1-score   support
       troll     0.5367    0.7302    0.6187       871
```

**Precision = 0.54** means 46% of troll predictions are WRONG (false positives).

### Root Cause Analysis:

| Finding | Data |
|---------|------|
| Troll-ONLY (pure sarcasm, no insult) | 2,344 samples (55.6% of all trolls) |
| Troll + insult overlap | 1,515 samples (36% of trolls) |
| Insult WITHOUT troll (confuser pool) | 3,481 samples |
| Pure insult (insult=1, troll=0, vulgar=0) | 1,344 samples |

**The model cannot distinguish INTENT:**
- **Troll** = mockery, sarcasm, ridicule, rhetorical questions ("তোমার বুদ্ধি কোথায়?")
- **Insult** = direct personal attack, explicit slurs ("খানকির পোলা")
- **Overlap** = mocking someone using insults (both labels = 1)

### Linguistic Evidence:
- **Troll-specific markers:** `?` (7.6× more than insult-only), `!` (28.7×), `জবাব` (7.2×), `হাসি` (16.5×)
- **Insult-specific markers:** `খানকি` (78×), `মাগী` (53×), `চোদা` (47×), `মাদার` (40×)

The model sees the same Bengali words in both classes but can't detect the sarcastic TONE that distinguishes them.

---

## v5.1 Solution: 6 Components

### Component 1: Sarcasm Feature Engineering (6 features)

```python
def extract_sarcasm_features(text):
    """6 hand-crafted features that distinguish troll from insult."""
    # 1. Question mark density (trolls ask rhetorical questions)
    # 2. Exclamation density (trolls use emphatic sarcasm)
    # 3. Punctuation diversity (trolls use varied punctuation)
    # 4. Word repetition ratio (trolls repeat for emphasis/mockery)
    # 5. Average word length (slurs tend to be longer)
    # 6. Has explicit slur markers (insult-specific vocabulary)
    return [feat_question, feat_excl, feat_punct_div,
            feat_repetition, feat_wlen, has_slur]
```

**Why it works:** These features encode TONE information that the BiGRU struggles to learn from text alone. Feature 6 (slur detection) directly tells the model "this is an insult, not a troll" when explicit slurs are present.

### Component 2: Asymmetric Focal Loss (Per-Class Gamma)

```python
PER_CLASS_GAMMAS = [1.5, 2.0, 3.0, 1.5, 1.5]
#                  vulg  thr  TROLL  ins  neu
# Troll gets gamma=3.0 → focuses MUCH harder on troll mistakes

# Plus: Label correlation penalty
CORR_PAIRS = [(2, 3)]  # (troll_idx, insult_idx)
# Penalizes: predicting troll=1 when ground truth is insult=1, troll=0
```

**Why it works:** Standard focal loss (γ=2) treats all classes equally. But troll has a fundamentally different error pattern — it needs aggressive focus on the insult-vs-troll boundary. The correlation penalty explicitly encodes the known confusion pattern.

### Component 3: V51Model with Sarcasm Fusion + LabelInteractionLayer

```
text → tokenize
  ├─► word_emb[300-d, frozen FastText] → projection[300→128] ─┐
  └─► char_emb(24) → char_CNN → max-pool(96) ────────────────┤
                                                              │
  sarcasm_features(6) ─────────────────────────────────────────┤
                                                              │
  ├─► concat(224) → CNN(96×3) → BiGRU(96×2) → LayerNorm ─────┤
  │                                                            │
  └─► Attention + MaxPool + AvgPool + SarcasmFeats(6) → FC(128) → FC(5)
                                                                    │
                                                    LabelInteractionLayer
                                                                    │
                                                              final logits
```

**LabelInteractionLayer:** Learns pairwise label interactions. When the model predicts high insult + low sarcasm features, it SUPPRESSES troll. This explicitly models the known troll-insult confusion.

### Component 4: Troll-Specific Augmentation

```python
# Target: troll=1, insult=0 samples (the hardest 55.6%)
# Methods:
# 1. Word swap (preserve meaning, change structure)
# 2. Repetition insertion (mimics emphasis in trolling: "কি কি বলো?")
# 3. Middle-shuffle (keep first/last, randomize middle — preserves tone markers)
```

**Why specifically troll-ONLY:** These pure sarcasm samples are the hardest to learn because they lack explicit toxic vocabulary. More diverse versions of them help the model learn sarcasm patterns.

### Component 5: Bug Fixes from v5

| Fix | v5 (broken) | v5.1 (fixed) | Impact |
|-----|-------------|--------------|--------|
| Augmentation | Before split (leakage!) | **After split** | Cleaner eval |
| MIN_FREQ | 2 (only 9,284 vocab) | **1 (15K+ vocab)** | Lower OOV (14% → ~8%) |
| Preprocessing | Removes English | **Keeps English + digits** | Preserves bilingual cues |
| Dropout | 0.50 | **0.55** | Reduces +0.097 gap |
| SWA start | 65% (before unfreeze!) | **80% (after unfreeze)** | Clean weight averaging |
| Unfreeze epoch | 25 | **28** | Let label interaction learn first |
| Unfreeze LR | 0.1× | **0.05×** | Less catastrophic forgetting |

### Component 6: Training Configuration

```python
class Config:
    VOCAB_SIZE = 25000       # Full vocab (was only 9K in v5)
    MIN_FREQ = 1             # Keep all words (FastText handles OOV)
    DROPOUT_EMB = 0.40       # Higher (combat overfitting)
    DROPOUT = 0.55           # Higher
    UNFREEZE_AT_EPOCH = 28   # Later (let label interaction train)
    UNFREEZE_LR_FACTOR = 0.05  # More conservative
    LR = 1.2e-3             # Slightly lower (more stable)
    WEIGHT_DECAY = 2e-4     # Higher (regularization)
    EPOCHS = 40             # More epochs (more data + augmentation)
    SWA_START_FRAC = 0.80   # After unfreeze stabilizes
    PATIENCE = 10           # More patience
    CORRELATION_PENALTY = 0.15  # Troll-insult confusion penalty
```

---

## Parameter Budget (STRICT: under 10M)

| Component | Params | Notes |
|-----------|--------|-------|
| FastText embedding (25K × 300) | 7.5M | Frozen Phase 1 / Unfrozen Phase 2 |
| OR FastText embedding (9.3K × 300) | 2.79M | With MIN_FREQ=2 (v5 actual) |
| Projection 300 → 128 | 38K | |
| Character CNN | 18K | |
| Word CNN (3 × 96) | 194K | |
| BiGRU (96 × 2 layers) | 389K | |
| LayerNorm | 384 | NEW |
| Attention | 37K | |
| Sarcasm fusion | ~1K | 6 extra input features |
| LabelInteractionLayer | ~1K | NEW: pairwise interactions |
| Classifier (FC1 + FC2) | ~19K | |
| **Total (with 9.3K vocab)** | **~3.5M** | **UNDER 10M ✓** |
| **Total (with 25K vocab)** | **~8.2M** | **UNDER 10M ✓** |

---

## T4x2 Parallel Mode

### Implementation:
```python
NUM_GPUS = torch.cuda.device_count()  # = 2
EFFECTIVE_BATCH = 64 * 2  # = 128

model = V51Model(cfg, embed_matrix, VOCAB_SIZE, CHAR_VOCAB_SIZE)
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
model = model.to(device)

# Access base model for state_dict:
base_model = model.module if hasattr(model, 'module') else model
```

### Timing (from v5 actual):
- 93 training batches per epoch (128 samples/batch)
- ~4 seconds per epoch on T4×2
- 40 epochs × 4s = ~3 minutes total

---

## Training & Validation Curves (KEY DELIVERABLE)

### 2×2 Subplot Figure (14×10 inches):

```
┌────────────────────────┬────────────────────────┐
│  [0,0] LOSS CURVES     │  [0,1] MACRO-F1        │
│  - Train loss (blue)   │  - Train F1 (blue)     │
│  - Val loss (red)      │  - Val F1 (red)        │
│  - Unfreeze line (grn) │  - Best epoch (gold)   │
│  - Phase 1 | Phase 2   │  - Unfreeze line (grn) │
├────────────────────────┼────────────────────────┤
│  [1,0] PER-CLASS F1    │  [1,1] LR SCHEDULE     │
│  - vulgar (red)        │  - Warmup → Cosine     │
│  - threat (orange)     │  - → SWA (constant)    │
│  - TROLL (green,bold)  │  - Unfreeze step (red) │
│  - insult (blue)       │  - Log scale           │
│  - neutral (purple)    │                        │
└────────────────────────┴────────────────────────┘
```

Plus per-epoch printing:
```
Ep 01 [  Frozen] | Loss 0.24/0.22 | F1 0.39/0.56 | Gap -0.17 | Troll:0.40 | LR 5e-4 | 4s *BEST*
Ep 10 [  Frozen] | Loss 0.14/0.14 | F1 0.74/0.75 | Gap -0.01 | Troll:0.63 | LR 1e-3 | 4s *BEST*
...
Ep 28 [Unfrozen] | Loss 0.11/0.12 | F1 0.82/0.80 | Gap +0.02 | Troll:0.72 | LR 6e-5 | 5s *BEST*
```

---

## Expected Results

| Metric | v5 (actual) | v5.1 (projected) | Improvement |
|--------|-------------|------------------|-------------|
| **Macro-F1** | 0.7791 | **0.80-0.82** | +0.02-0.04 |
| Vulgar F1 | 0.8563 | 0.86+ | maintained |
| Threat F1 | 0.8898 | 0.89+ | maintained |
| **Troll F1** | **0.6187** | **0.70-0.74** | **+0.08-0.12** |
| Insult F1 | 0.7619 | 0.77+ | slight boost |
| Neutral F1 | 0.7689 | 0.77+ | maintained |
| Train-Val Gap | +0.072 | **<0.05** | less overfitting |
| ROC-AUC | 0.9240 | 0.93+ | improved |
| Total Params | 3.53M | 3.5-8.2M | **UNDER 10M ✓** |

---

## Complete Notebook Sections (v5.1)

### Section 1 — Title & Introduction (Markdown)
### Section 2 — Setup & Imports (Code)
### Section 3 — Configuration with Troll Fixes (Code)
### Section 4 — Load & Clean Dataset (Code)
### Section 5 — Stratified Split FIRST (Code) ← FIX: split before augmentation
### Section 6 — Threat-Class Augmentation on TRAIN ONLY (Code)
### Section 7 — Troll-Class Augmentation on TRAIN ONLY (Code) ← NEW
### Section 8 — EDA (Code)
### Section 9 — Text Preprocessing (Code) ← FIX: keep English
### Section 10 — Vocabularies (Code) ← FIX: MIN_FREQ=1
### Section 11 — Load FastText (Code)
### Section 12 — Sarcasm Feature Engineering (Code) ← NEW
### Section 13 — Enhanced Dataset & DataLoaders (Code) ← includes sarcasm features
### Section 14 — V51Model Architecture (Code) ← NEW: sarcasm fusion + LabelInteraction + LayerNorm
### Section 15 — Asymmetric Focal Loss & Training Setup (Code) ← NEW: per-class gamma + correlation penalty
### Section 16 — Training Loop: Two-Phase (Code) ← FIX: SWA after unfreeze
### Section 17 — Training & Validation Curves (Code) ← KEY DELIVERABLE
### Section 18 — Threshold Tuning (Code)
### Section 19 — Final Test Evaluation (Code)
### Section 20 — Ensemble Strategy (Code)
### Section 21 — Save Model & Summary (Code)
### Section 22 — Conclusion & Version Comparison (Markdown)

---

## How to Run on Kaggle

1. Create new Kaggle notebook
2. Add the 15K balanced CSV dataset as input
3. Add FastText Bengali vectors dataset (or enable internet)
4. **Select accelerator: GPU T4 × 2**
5. Upload `v5_troll_fix.py` or paste into cells
6. Run all → ~4 minutes training
7. Check troll F1 in per-class report

---

## File Structure

```
bengali_cyber/
├── bengali-cyberbullying-lightweight-v4.ipynb    # v4 baseline (F1=0.7323)
├── bengali-cyberbullying-v5-t4x2.ipynb          # v5 with outputs (F1=0.7791)
├── v5_troll_fix.py                              # v5.1 troll fix (target 0.82+)
├── v5_notebook_plan.md                          # THIS FILE
├── v5_train.py                                  # Original v5 script
└── combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv  # 15K dataset
```
