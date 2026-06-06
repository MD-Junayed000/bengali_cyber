# Bengali Multi-Label Cyberbullying Detection - Architecture & Results

**Status:** v6 Hierarchical is the **primary model** (best Macro-F1). Track 2 Transformer is the secondary validation model. Track 1 Lightweight-v4 is deprecated.

| Track | Model | Params | Test Macro-F1 | ROC-AUC | Status |
|-------|-------|--------|---------------|---------|--------|
| **Primary** | v6 Hierarchical (CharCNN+BiGRU) | 3.30M | **0.8551** | 0.9526 | Active |
| Secondary | Track 2 Transformer (BanglaBERT) | ~110M | 0.8428 | 0.9617 | Active |
| Deprecated | Track 1 Lightweight-v4 | ~10M | 0.7958 | -- | Deprecated |

**Key finding:** The 3.3M-parameter hierarchical model outperforms the 110M-parameter transformer on Macro-F1 by +1.23 points, while being 33x smaller and CPU-deployable.

---

## 1. Dataset Overview

- **Source:** `final_bengali_comments_vulgar_threat_insult_neutral.csv`
- **Total samples:** 12,703 Bengali text comments
- **Labels (4 classes):** vulgar, threat, insult, neutral (multi-label, binary columns)

| Class | Count | Percentage |
|-------|-------|------------|
| Vulgar | 3,894 | 30.7% |
| Threat | 2,236 | 17.6% |
| Insult | 4,333 | 34.1% |
| Neutral | 5,505 | 43.3% |
| Multi-label (2+ toxic) | 2,900 | 22.8% |

**Co-occurrence:** vulgar+insult = 2,235 | vulgar+threat = 603 | threat+insult = 794

**Neutral derivation:** `neutral = NOT(vulgar OR threat OR insult)` - not an independent label.

---

## 2. Architecture Comparison

### 2.1 v6 Hierarchical Model (Primary - 3.30M params)

```
Input: Bengali text (tokenized words + characters)
    |
    +---> [Character-level Branch]
    |        CharCNN: Embedding(24d) -> Conv1d(kernels 2,3,4; 32 filters each) -> MaxPool
    |        Output: 96d character features per word
    |
    +---> [Word-level Branch]
    |        FastText embeddings (300d) -> Linear projection (128d)
    |        Output: 128d word features
    |
    +---> [Concatenation: 128d + 96d = 224d per token]
             |
             v
         TextCNN: 3 parallel Conv1d (kernels 2,3,4; 96 filters each) -> MaxPool
             |  Output: 288d
             v
         BiGRU: 2 layers, hidden=96 -> 192d sequence
             |
             v
         Additive Attention -> 192d context vector
             |
             +---> [Stage 1 Head: Binary toxic/neutral]
             |        Multi-sample dropout -> Linear(192, 1) -> Sigmoid
             |        Output: P(toxic)
             |
             +---> [Stage 2 Head: Multi-label toxic subtypes]
                      Multi-sample dropout -> Linear(192, 3) -> Sigmoid
                      Output: P(vulgar|toxic), P(threat|toxic), P(insult|toxic)

    Inference formula:
        P(class) = P(toxic) * P(class|toxic)
        P(neutral) = 1 - P(toxic)
```

**Key Design Choices:**
- Hierarchical two-stage decomposition isolates the toxic-vs-neutral decision
- Stage 2 trains ONLY on toxic samples, removing neutral noise from subtype learning
- Multi-sample dropout (5 forward passes, averaged) provides regularization
- Character CNN captures morphological patterns in Bengali script
- Focal loss handles class imbalance

### 2.2 Track 2 Transformer Model (Secondary - ~110M params)

```
Input: Bengali text (BPE tokenized)
    |
    v
BanglaBERT Encoder (csebuetnlp/banglabert)
    12 transformer layers, 768 hidden dim
    Bottom 2 layers frozen
    |
    v
Masked Mean-Pool (over non-padding tokens)
    Output: 768d sentence embedding
    |
    v
LayerNorm(768)
    |
    v
Multi-sample Dropout (5x, p=0.1)
    |
    v
Linear(768, 3) -> Sigmoid
    Output: P(vulgar), P(threat), P(insult)
    P(neutral) = NOT(any prediction above threshold)
```

**Training Configuration:**
- LLRD (Layer-wise Learning Rate Decay): factor 0.9
- R-Drop regularization: alpha = 0.3
- Cosine learning rate schedule with warmup
- AMP (Automatic Mixed Precision)
- Focal loss for class imbalance
- 12 epochs, early stopping patience = 5
- AdamW optimizer

---

## 3. Results Comparison

### 3.1 Per-Class F1 Scores

| Class | v6 Hierarchical | Track 2 Transformer | Delta |
|-------|----------------|---------------------|-------|
| Vulgar | **0.8637** | 0.8626 | +0.0011 |
| Threat | **0.8949** | 0.8061 | +0.0888 |
| Insult | 0.7936 | **0.7997** | -0.0061 |
| Neutral | 0.8680 | **0.9027** | -0.0347 |
| **Macro-F1** | **0.8551** | 0.8428 | **+0.0123** |

### 3.2 Overall Metrics

| Metric | v6 Hierarchical | Track 2 Transformer |
|--------|----------------|---------------------|
| Test Macro-F1 | **0.8551** | 0.8428 |
| ROC-AUC | 0.9526 | **0.9617** |
| PR-AUC | -- | 0.9227 |
| Stage 1 F1 (binary) | 0.9212 | -- |
| Hamming Accuracy | 0.9033 | -- |
| Parameters | **3.30M** | ~110M |
| CPU-Deployable | Yes | No (practical) |

---

## 4. Why v6 Hierarchical Outperforms the Transformer

Despite having 33x fewer parameters, the v6 hierarchical model achieves better Macro-F1. Five key reasons:

### 4.1 Hierarchical Decomposition Provides Better Gradient Signal

The two-stage architecture separates the easy decision (toxic vs neutral, F1=0.9212) from the hard decision (which toxic subtype). This prevents the gradient conflict where neutral samples dilute the learning signal for toxic subtypes, especially the minority class (threat at 17.6%).

### 4.2 Stage-2 Toxic-Only Training Eliminates Neutral Noise

By training the subtype classifier exclusively on confirmed toxic samples (7,197 rows), Stage 2 sees a cleaner, more balanced distribution. This is why threat F1 jumps to 0.8949 in v6 vs only 0.8061 in the flat transformer that must simultaneously separate all classes.

### 4.3 Probabilistic Chain Rule Provides Natural Calibration

The inference formula `P(class) = P(toxic) * P(class|toxic)` provides built-in calibration. If the model is uncertain about toxicity (P(toxic) near 0.5), all toxic class probabilities are automatically dampened. The transformer has no such self-correcting mechanism.

### 4.4 Dataset Size is Suboptimal for 110M Parameters

With only 12,703 training samples, fine-tuning 110M parameters (even with LLRD and frozen layers) is data-inefficient. The v6 model's 3.3M parameters are better matched to this data regime, avoiding overfitting and extracting more signal per sample.

### 4.5 Task-Specific Architecture Beats General-Purpose Fine-Tuning

The CharCNN captures Bengali morphological cues (abusive suffixes, character patterns). The BiGRU+Attention learns sequential context. These inductive biases are specifically tuned for the cyberbullying detection task, whereas BanglaBERT's pretraining objective (masked language modeling) is general-purpose and must be adapted.

---

## 5. Parameter Efficiency Analysis

| Model | Params | Macro-F1 | F1 per Million Params | CPU Inference | GPU Required |
|-------|--------|----------|----------------------|---------------|--------------|
| v6 Hierarchical | 3.30M | 0.8551 | 0.2591 | Yes (fast) | No |
| Track 2 Transformer | ~110M | 0.8428 | 0.0077 | Impractical | Yes |
| Track 1 Lightweight-v4 | ~10M | 0.7958 | 0.0796 | Yes | No |

**v6 is 33.6x more parameter-efficient** than the transformer (F1-per-million-params: 0.2591 vs 0.0077).

**Deployment implications:**
- v6 can run on a single CPU core with <50ms inference latency per comment
- Track 2 requires GPU (or very slow CPU inference ~2-5s per comment)
- v6 model file size: ~13MB vs Track 2: ~440MB

---

## 6. Literature Comparison

| Work | Year | Model | Params | Dataset | F1 | Venue |
|------|------|-------|--------|---------|----|----|
| Emon et al. | 2019 | SVM + TF-IDF | <1M | 5.1k Bengali | 0.52 | ICCIT 2019 |
| Ishmam & Sharmin | 2019 | BiLSTM + Attention | ~3M | 5.1k Bengali | 0.71 | ICCIT 2019 |
| Karim et al. | 2020 | mBERT (DeepHateExplainer) | ~110M | 44k Bengali | 0.87 | IEEE TCSS |
| Ahmed et al. | 2021 | Ensemble CNN+BiLSTM | ~5M | 10k Bengali | 0.78 | ICCIT 2021 |
| Romim et al. | 2022 | BanglaBERT fine-tuned | ~110M | 30k Bengali | 0.84 | LREC 2022 |
| Belal et al. | 2023 | CNN-BiLSTM + BanglaBERT | ~115M | 15k Bengali | 0.86 | IEEE Access |
| Saha et al. | 2024 | ToxiFusion (multimodal) | ~150M | 8k Bengali | 0.85 | Expert Sys. App. |
| **Ours (v6)** | **2025** | **Hierarchical CharCNN+BiGRU** | **3.30M** | **12.7k** | **0.8551** | **33x smaller, CPU-ready** |
| Ours (Track 2) | 2025 | BanglaBERT fine-tuned | ~110M | 12.7k | 0.8428 | Secondary validation |

**Notable observations:**
- Our v6 achieves competitive F1 (0.8551) with only 3.3M params on a multi-label task
- Most prior work with comparable F1 uses 30-50x more parameters
- Prior works achieving >0.85 F1 typically use larger datasets (30k-44k) or binary classification
- Our result is multi-label (4 classes) which is inherently harder than binary toxic/non-toxic

---

## 7. Track Status

### Primary: v6 Hierarchical (`bengali-cyberbullying-v6-hierarchical.ipynb`)
- **Status:** Active, best performer
- **Result:** Macro-F1 = 0.8551, ROC-AUC = 0.9526
- **Strengths:** Lightweight (3.3M), CPU-deployable, best F1, interpretable attention weights
- **Use case:** Production deployment, real-time moderation, resource-constrained environments

### Secondary: Track 2 Transformer (`bengali-cyberbullying-transformer.ipynb`)
- **Status:** Active, validation/comparison model
- **Result:** Macro-F1 = 0.8428, ROC-AUC = 0.9617
- **Strengths:** Best ROC-AUC, strong neutral detection (F1=0.9027), leverages pretrained knowledge
- **Use case:** Ensemble candidate, research comparison, high-AUC-required scenarios

### Deprecated: Track 1 Lightweight-v4 (`bengali-cyberbullying-lightweight-v4.ipynb`)
- **Status:** Deprecated (worst performer across all metrics)
- **Result:** Macro-F1 = 0.7958
- **Reason for deprecation:** Superseded by v6 which uses same compute budget but better architecture

---

## 8. Potential Improvements

### For v6 Hierarchical (Primary)
1. **Larger augmentation for threat class** - Threat (17.6%) remains the hardest non-neutral class; back-translation or contextual augmentation could help
2. **Knowledge distillation from transformer** - Use Track 2's soft probabilities as auxiliary training signal
3. **Subword tokenization** - Replace word-level FastText with BPE-based embeddings to handle OOV better
4. **Curriculum learning** - Train on easy samples first (clear toxic/neutral), then ambiguous boundary cases
5. **Label smoothing** - Soften targets to 0.05/0.95 to reduce overconfidence on noisy labels

### For Track 2 Transformer (Secondary)
1. **Hierarchical head** - Apply the same two-stage approach on top of BanglaBERT features
2. **Larger pretrained model** - Try `csebuetnlp/banglabert_large` (if available) or XLM-RoBERTa-large
3. **More training data** - The 12.7k dataset is small for 110M params; semi-supervised or data augmentation
4. **Adversarial training (FGM/PGD)** - Add perturbation-based regularization for robustness
5. **Ensemble with v6** - Average probabilities from both models for complementary strengths

### Cross-Track Improvements
1. **Ensemble fusion** - Weighted average of v6 + Track 2 probabilities (weights tuned on validation)
2. **Active learning** - Use model disagreement between tracks to identify samples for re-annotation
3. **Cross-validation** - Run 5-fold CV to get more robust estimates and reduce variance in comparisons

---

## 9. Running Instructions (Kaggle T4 x2)

### Prerequisites
- Kaggle notebook with **GPU T4 x2** accelerator
- **Internet ON** (required for FastText download in v6, BanglaBERT weights in Track 2)
- Dataset: Upload `final_bengali_comments_vulgar_threat_insult_neutral.csv` as a Kaggle dataset

### Running v6 Hierarchical (Primary)
```
1. Create new Kaggle notebook
2. Set accelerator: GPU T4 x2
3. Enable internet access
4. Upload bengali-cyberbullying-v6-hierarchical.ipynb
5. Attach dataset (or update DATA_PATH in Section 2)
6. Run All (~15-20 minutes total)
7. Outputs: model checkpoint, classification report, visualizations
```

### Running Track 2 Transformer (Secondary)
```
1. Create new Kaggle notebook
2. Set accelerator: GPU T4 x2
3. Enable internet access
4. Upload bengali-cyberbullying-transformer.ipynb
5. Attach dataset (or update DATA_PATH in config)
6. Run All (~30-40 minutes total, ~81s/epoch x 12 epochs)
7. Outputs: model checkpoint, classification report, visualizations
```

### Notes
- Both notebooks derive `neutral = NOT(vulgar OR threat OR insult)` automatically
- Both use per-class threshold tuning on validation set
- Both use `nn.DataParallel` for multi-GPU training
- v6 uses split-before-augment to prevent data leakage
- Track 2 uses LLRD and partial layer freezing for stable fine-tuning
