# Bengali Multi-Label Cyberbullying Detection — v5 (T4x2 Parallel)

## Lightweight Pretrained-Embedding Model (~8.5M total / ~0.75M trainable, UNDER 10M)

**5-label multi-label classification:** `vulgar`, `threat`, `troll`, `insult`, `neutral`

---

## Why v5 Exists

v4 achieved **macro-F1 = 0.7323** with a +0.058 train-val gap on ~10.6K samples.
v5 leverages:
1. **New 15K balanced dataset** (+43% more training data)
2. **T4x2 parallel GPU training** (DataParallel → larger effective batch, faster epochs)
3. **Focal Loss** for the severely imbalanced `threat` class (6.36:1 neg/pos)
4. **Embedding fine-tuning** in the final phase (unfreeze last 30% of training)
5. **Cosine Annealing with Warm Restarts** for better convergence
6. **Increased vocab** (25K→30K) to capture the richer vocabulary

| Version | Macro-F1 | Train-val gap | Dataset | Key Change |
|---------|----------|---------------|---------|------------|
| v3 | 0.6866 | +0.166 | 10.6K | No semantic priors |
| v4 | 0.7323 | +0.058 | 10.6K | Frozen FastText |
| **v5** | **target ≥0.78** | target <0.06 | **15.2K** | +data, focal loss, unfreeze, T4x2 |

### Parameter Budget (STRICT: under 10M)

| Component | Params | Trainable? |
|-----------|--------|------------|
| FastText embedding 25K × 300 | 7.5M | Phase 1: No / Phase 2: Yes (last 5 epochs) |
| Projection 300 → 128 | 38K | Yes |
| Character CNN | 18K | Yes |
| Word CNN (3 kernels × 96 filters) | 194K | Yes |
| 2-layer BiGRU (hidden=96) | 389K | Yes |
| Attention + classifier | 56K | Yes |
| **Total** | **~8.2M** | **Phase 1: ~0.69M / Phase 2: ~8.2M** |

---

## Complete Notebook Sections

### Section 1 — Setup & Imports
- Install `iterative-stratification`
- Import PyTorch, numpy, pandas, sklearn, matplotlib
- Set seeds for reproducibility
- Detect GPUs (expect 2x T4)

### Section 2 — Configuration
```python
class Config:
    # ----- data -----
    DATA_PATH = '/kaggle/input/your-dataset/combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv'
    TEXT_COL = 'text'
    LABEL_COLS = ['vulgar', 'threat', 'troll', 'insult', 'neutral']
    NUM_CLASSES = 5
    TOXIC_COLS = ['vulgar', 'threat', 'troll', 'insult']

    # ----- preprocessing -----
    MIN_WORDS = 2

    # ----- vocab -----
    VOCAB_SIZE = 25_000          # Still 25K to stay under 10M total
    MIN_FREQ = 2                 # Bump from 1→2 (more data = can afford higher min)
    MAX_LEN = 80

    CHAR_VOCAB_SIZE = 250
    MAX_CHAR_PER_WORD = 16
    CHAR_EMBED_DIM = 24
    CHAR_CNN_FILTERS = 32
    CHAR_KERNELS = (2, 3, 4)

    # ----- pretrained embeddings -----
    USE_PRETRAINED = True
    FASTTEXT_DIM = 300
    FASTTEXT_URL = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.bn.300.vec.gz'
    FREEZE_EMBEDDING = True      # Phase 1: frozen
    UNFREEZE_AT_EPOCH = 25       # Phase 2: unfreeze at epoch 25 of 35
    UNFREEZE_LR_FACTOR = 0.1    # 10x smaller LR for embeddings when unfrozen
    PROJECTION_DIM = 128

    # ----- split (stratified 70/10/20) -----
    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.10
    TEST_FRAC = 0.20

    # ----- model -----
    CNN_FILTERS = 96
    CNN_KERNELS = (2, 3, 4)
    GRU_HIDDEN = 96
    GRU_LAYERS = 2
    DROPOUT_EMB = 0.35           # Slightly less dropout (more data = less overfitting)
    DROPOUT = 0.5                # Reduced from 0.6
    NUM_DROPOUT_SAMPLES = 5

    # ----- training (T4x2 parallel) -----
    BATCH_SIZE_PER_GPU = 64      # 64 per GPU × 2 GPUs = 128 effective
    NUM_GPUS = 2
    EFFECTIVE_BATCH_SIZE = 128
    EPOCHS = 35                  # More epochs (more data needs more time)
    LR = 1.5e-3                  # Slightly higher (larger batch)
    WEIGHT_DECAY = 1e-4
    WARMUP_RATIO = 0.08
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.05

    # ----- focal loss (NEW in v5) -----
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0            # Focus on hard examples
    FOCAL_ALPHA_THREAT = 0.85    # Extra weight for underrepresented threat class

    # ----- augmentation -----
    WORD_DROPOUT_P = 0.18
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.5

    # ----- SWA -----
    USE_SWA = True
    SWA_START_FRAC = 0.65
    SWA_LR = 2e-4

    # ----- early stopping -----
    EARLY_STOP_ON = 'val_loss'
    PATIENCE = 8                 # More patience (more epochs)

    # ----- evaluation -----
    DEFAULT_THRESHOLD = 0.5
    THRESHOLD_GRID = np.arange(0.10, 0.85, 0.01)
```

### Section 3 — Multi-GPU Setup
```python
import torch.nn as nn

# Detect available GPUs
NUM_GPUS = torch.cuda.device_count()
print(f'Available GPUs: {NUM_GPUS}')
for i in range(NUM_GPUS):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')

DEVICE = torch.device('cuda:0')  # Primary device

# DataParallel wrapper (applied after model creation)
def parallelize_model(model):
    if NUM_GPUS > 1:
        print(f'Using DataParallel across {NUM_GPUS} GPUs')
        model = nn.DataParallel(model)
    return model.to(DEVICE)
```

### Section 4 — Load & Clean Dataset (15K)
- Load the new CSV: `combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv`
- Fix 6 neutral+toxic contradictions
- Merge duplicates (64 redundant rows)
- Drop 98 single-word samples
- Expected final size: ~15,000+ rows

### Section 5 — EDA
- Per-class positive counts bar chart
- Labels-per-comment distribution
- Word count histogram with MAX_LEN line
- **NEW:** Class imbalance visualization (neg/pos ratio per class)
- **NEW:** Co-occurrence heatmap

### Section 6 — Text Preprocessing
- Same as v4 (URL, mention, hashtag, emoji removal; Bengali/English character filtering)

### Section 7 — Stratified Split (70/10/20)
- ~10,600 train / ~1,500 val / ~3,000 test
- Verify per-class positive rates are balanced across splits

### Section 8 — Vocabularies
- Word vocab: min_freq=2, cap=25K (to keep total params < 10M)
- Char vocab: 250
- Report OOV rates

### Section 9 — Load Pretrained FastText
- Same streaming approach as v4
- Report coverage (expected >88% with richer vocab)

### Section 10 — Dataset & DataLoaders (T4x2 optimized)
```python
# Key change: batch_size = per_gpu_batch × num_gpus
effective_bs = cfg.BATCH_SIZE_PER_GPU * cfg.NUM_GPUS  # 128

train_loader = DataLoader(train_ds, batch_size=effective_bs,
                          shuffle=True, num_workers=4, pin_memory=True,
                          drop_last=True)  # drop_last for DataParallel consistency
val_loader = DataLoader(val_ds, batch_size=effective_bs,
                        shuffle=False, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=effective_bs,
                         shuffle=False, num_workers=4, pin_memory=True)
```

### Section 11 — Model Architecture (Same as v4, under 10M)
- Identical V4Model architecture
- Verify total params < 10M
- Apply `nn.DataParallel` for T4x2

### Section 12 — Focal Loss (NEW in v5)
```python
class FocalBCELoss(nn.Module):
    """Per-class focal loss with asymmetric alpha for imbalanced classes."""
    def __init__(self, gamma=2.0, alpha=None, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # per-class alpha tensor
        self.pos_weight = pos_weight
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight)

        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * bce

        if self.alpha is not None:
            loss = loss * self.alpha.unsqueeze(0)

        return loss.mean()
```

### Section 13 — Training Setup (Two-Phase LR)
```python
# Phase 1: Frozen embeddings, normal LR (epochs 1-24)
# Phase 2: Unfrozen embeddings, embedding LR = 0.1× main LR (epochs 25-35)

# Optimizer with parameter groups (for phase 2)
def create_optimizer(model, cfg, phase=1):
    if phase == 1:
        params = [p for p in model.parameters() if p.requires_grad]
        return AdamW(params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    else:
        # Phase 2: differential learning rates
        emb_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'word_embedding' in name:
                param.requires_grad = True
                emb_params.append(param)
            elif param.requires_grad:
                other_params.append(param)
        return AdamW([
            {'params': other_params, 'lr': cfg.LR * 0.5},
            {'params': emb_params, 'lr': cfg.LR * cfg.UNFREEZE_LR_FACTOR},
        ], weight_decay=cfg.WEIGHT_DECAY)

# Scheduler: CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-5)
```

### Section 14 — Training Loop (with parallel GPUs + two phases)
```python
# Key features:
# 1. DataParallel across 2x T4
# 2. Phase 1 (epochs 1-24): frozen embeddings, focal loss
# 3. Phase 2 (epochs 25-35): unfrozen embeddings, reduced LR
# 4. Mixup augmentation
# 5. Multi-sample dropout (5 forward passes during training)
# 6. SWA in final epochs
# 7. Track train/val loss AND accuracy curves per epoch

history = {
    'train_loss': [], 'val_loss': [],
    'train_macro_f1': [], 'val_macro_f1': [],
    'train_accuracy': [], 'val_accuracy': [],  # NEW: subset accuracy
    'val_per_class_f1': [],
    'lr': [],
    'phase': [],  # track which phase each epoch was in
}
```

### Section 15 — Training & Validation Curves (KEY DELIVERABLE)
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss curves
axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
axes[0,0].axvline(cfg.UNFREEZE_AT_EPOCH, color='green', ls='--', label='Unfreeze')
axes[0,0].set_title('Training & Validation Loss')
axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

# Plot 2: Macro-F1 curves (accuracy proxy for multi-label)
axes[0,1].plot(epochs, history['train_macro_f1'], 'b-', label='Train Macro-F1')
axes[0,1].plot(epochs, history['val_macro_f1'], 'r-', label='Val Macro-F1')
axes[0,1].axvline(cfg.UNFREEZE_AT_EPOCH, color='green', ls='--', label='Unfreeze')
axes[0,1].set_title('Training & Validation Macro-F1')
axes[0,1].legend(); axes[0,1].grid(alpha=0.3)

# Plot 3: Per-class val F1 over epochs
for i, c in enumerate(cfg.LABEL_COLS):
    axes[1,0].plot(epochs, per_cls[:, i], 'o-', label=c)
axes[1,0].set_title('Per-Class Val F1')
axes[1,0].legend(); axes[1,0].grid(alpha=0.3)

# Plot 4: Learning rate schedule
axes[1,1].plot(epochs, history['lr'], 'g-')
axes[1,1].axvline(cfg.UNFREEZE_AT_EPOCH, color='red', ls='--', label='Phase 2')
axes[1,1].set_title('Learning Rate Schedule')
axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('v5_training_curves.png', dpi=150)
plt.show()
```

### Section 16 — Threshold Tuning on Validation Set
- Same grid search as v4

### Section 17 — Final Test-Set Evaluation
- Macro/Micro/Weighted/Samples F1
- Hamming Loss
- ROC-AUC, PR-AUC
- Per-class classification report
- Confusion matrices

### Section 18 — Evaluation Visualizations
- Per-class F1 bar chart
- ROC curves per class
- PR curves per class
- Confusion matrix heatmaps

### Section 19 — Overfitting Diagnostic
```python
print('OVERFITTING DIAGNOSTIC')
print(f'Final train F1: {final_tr:.4f}')
print(f'Final val   F1: {final_va:.4f}')
print(f'Gap: {gap:+.4f}')
print(f'v4 gap: +0.058 → v5 gap: {gap:+.3f}')
```

### Section 20 — Error Analysis
- Top misclassified examples per class
- Attention visualization on failures

### Section 21 — Save Model & Summary
- Save checkpoint, thresholds, config, history
- Export summary JSON

### Section 22 — Version Comparison Table
```
| Version | Data  | Params  | Macro-F1 | Gap    | Key Feature            |
|---------|-------|---------|----------|--------|------------------------|
| v3      | 10.6K | ~1.5M  | 0.6866   | +0.166 | Random embeddings      |
| v4      | 10.6K | ~8.2M  | 0.7323   | +0.058 | Frozen FastText        |
| v5      | 15.2K | ~8.2M  | ≥0.78?   | <0.06  | +Data, Focal, Unfreeze |
```

---

## T4x2 Parallel Training — Technical Details

### How DataParallel Works on 2x T4:
1. Model is replicated on both GPUs
2. Each mini-batch is split: 64 samples → GPU0, 64 samples → GPU1
3. Forward pass runs simultaneously on both GPUs
4. Gradients are gathered on GPU0, averaged, and model updated
5. ~1.7-1.8× speedup (not perfect 2× due to communication overhead)

### Practical Code:
```python
# After model creation
model = V5Model(cfg, word_vocab_size, char_vocab_size, pretrained_emb)
model = nn.DataParallel(model)  # Wrap for multi-GPU
model = model.to(DEVICE)

# Access underlying model (for state_dict, etc.)
base_model = model.module if hasattr(model, 'module') else model

# For unfreezing embeddings in phase 2:
base_model.word_embedding.weight.requires_grad = True
# Must re-create optimizer with new param groups
```

### Timing Expectations (T4x2 vs T4x1):
- v4 on T4x1: ~4.4s per epoch (7,404 train samples)
- v5 on T4x2: ~4-5s per epoch (10,600 train samples) — larger data but 2 GPUs
- Total training: 35 epochs × 5s ≈ ~3 minutes

---

## Key Differences from v4 → v5

| Aspect | v4 | v5 |
|--------|----|----|
| Dataset | 10,594 samples | ~15,100 samples |
| GPUs | 1x T4 | **2x T4 (DataParallel)** |
| Batch size | 64 | **128 (64 per GPU)** |
| Loss | Smoothed BCE | **Focal Loss (γ=2)** |
| Embedding | Frozen forever | **Unfreeze at epoch 25** |
| Vocab min_freq | 1 | **2** |
| Epochs | 30 | **35** |
| Dropout | 0.6 | **0.5** (more data) |
| LR | 1e-3 | **1.5e-3** (larger batch) |
| Scheduler | Warmup+Cosine+SWA | **CosineWarmRestarts+SWA** |
| Patience | 6 | **8** |
| Total params | 8.2M (<10M ✓) | **8.2M (<10M ✓)** |
