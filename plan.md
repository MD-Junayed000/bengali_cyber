# Bengali Multi-Label Cyberbullying Detection — Implementation Plan (2026)

Two notebooks implement one leakage-free pipeline on the same dataset and splits. Both predict the
three toxic labels (`vulgar`, `threat`, `insult`) and **derive `neutral = NOT(any toxic)`**, then
tune per-class thresholds on validation.

- **`bengali-cyberbullying-v6-hierarchical.ipynb`** — **primary model**. A compact (3.30M-param)
  hierarchical CharCNN + FastText + BiGRU + Attention network. **Best result: Macro-F1 = 0.8551**,
  CPU-deployable.
- **`bengali-cyberbullying-transformer.ipynb`** — **Track 2**, a BanglaBERT (~110M) fine-tune with a
  hierarchical 2-stage head, used as a comparison/ranking model.

| Model | Params | Test Macro-F1 | ROC-AUC | CPU-deployable | Status |
|-------|--------|---------------|---------|----------------|--------|
| **v6 Hierarchical (primary)** | **3.30M** | **0.8551** | 0.9526 | Yes | Active |
| Track 2 BanglaBERT + hierarchical head | ~110M | 0.8343 | 0.9575 | No | Active (comparison) |

**Headline finding:** the 3.30M-parameter v6 model beats the 110M-parameter transformer on Macro-F1
by **+2.08 points** while being **33x smaller** and runnable on CPU.

---

## 1. Dataset

- **File:** `final_bengali_comments_vulgar_threat_insult_neutral.csv`
- **Rows:** ~12,700 Bengali comments (after light clean + dedup: ~12,500).
- **Labels:** multi-label binary columns `vulgar`, `threat`, `insult`; `neutral` is **derived**.

| Class | Approx. share |
|-------|---------------|
| vulgar | ~30.5% |
| threat | ~17.2% (rarest toxic class) |
| insult | ~34.0% |
| neutral (derived) | ~43.7% |

`is_toxic` (any of the three) ≈ 56% — this is the Stage-1 positive rate.

---

## 2. Architecture — v6 Hierarchical (primary, 3.30M params)

```
Input text (word ids + char ids)
        |
        +-- Char branch:  CharEmbed -> CharCNN (k=2,3,4) -> per-word char features
        +-- Word branch:  FastText cc.bn.300 (300d, frozen ep 1-24) -> projection
        |
        v   concat(word, char) per token
   TextCNN (parallel k=2,3,4)  +  BiGRU (2 layers, hidden 96 -> 192d)
        |
   Additive Attention -> context vector
        |
   +----+--------------------------+
   |                               |
   v                               v
 Stage 1 head                   Stage 2 head
 Linear -> P(toxic)             Linear -> P(vulgar|tox), P(threat|tox), P(insult|tox)
 (binary toxic vs neutral)      (multi-sample dropout; trained on TOXIC rows only)

Inference:  P(class)   = P(toxic) * P(class|toxic)
            P(neutral) = 1 - P(toxic)
```

**Training:** focal loss with per-stage `pos_weight`; combined loss `L = a*Stage1 + b*Stage2`;
frozen embeddings epochs 1-24 then unfrozen 25-35; SWA over the post-unfreeze plateau; mixup;
multi-sample dropout; cosine schedule with warmup; per-class threshold tuning on validation.

**Why it works:** Stage 1 is an easy boundary (binary F1 = 0.9212) that absorbs neutral noise;
Stage 2 trains only on toxic rows, so minority-class gradients (threat at ~17%) are not diluted by
the neutral majority; the probabilistic chain self-calibrates and suppresses false positives.

---

## 3. Architecture — Track 2 BanglaBERT (comparison, ~110M params)

```
Input text (WordPiece)
        |
        v
BanglaBERT encoder (csebuetnlp/banglabert, ELECTRA-base; bottom 2 layers frozen)
        |
   masked mean-pool -> shared LayerNorm -> multi-sample dropout (5x)
        |
   +----+---------------------------+
   |                                |
   v                                v
 Stage 1 head                    Stage 2 head
 Linear(768 -> 1) = P(toxic)     Linear(768 -> 3) = P(subtype | toxic)

Inference:  P(class) = P(toxic) * P(subtype|toxic);  P(neutral) = 1 - P(toxic)
```

**Training:** combined `a*Stage1 + b*Stage2` focal loss (Stage-2 masked to toxic rows), LLRD
(factor 0.9), R-Drop (alpha 0.3), cosine schedule with warmup, AMP, `nn.DataParallel`, early
stopping; per-class threshold tuning on validation.

**Honest ablation — does the hierarchical head transfer to the transformer? No.**
- Flat mean-pool head (previous baseline): **0.8428**
- Hierarchical 2-stage head (current): **0.8343** (−0.0085)

A pretrained encoder already encodes the toxic/neutral and subtype signals jointly. Forcing a hard
two-stage split discards information the encoder relied on (most visibly hurting `threat`), so the
decomposition that helps the from-scratch v6 model does **not** transfer to BanglaBERT. This is a
useful negative result: small-model architectural tricks are not universally portable.

---

## 4. Results

### 4.1 Overall

| Metric | v6 Hierarchical | Track 2 (hier head) |
|--------|-----------------|---------------------|
| Test Macro-F1 | **0.8551** | 0.8343 |
| Micro-F1 | 0.8528 | 0.8458 |
| ROC-AUC | 0.9526 | **0.9575** |
| PR-AUC | 0.9185 | 0.9146 |
| Stage-1 binary F1 | 0.9212 | — |
| Hamming accuracy | 0.9033 | ~0.9018 |
| Parameters | **3.30M** | ~110M |

### 4.2 Per-class F1

| Class | v6 Hierarchical | Track 2 | Winner |
|-------|-----------------|---------|--------|
| vulgar | **0.8637** | 0.8579 | v6 (+0.0058) |
| threat | **0.8949** | 0.7883 | **v6 (+0.1066)** |
| insult | **0.7936** | 0.7900 | v6 (+0.0036) |
| neutral | 0.8680 | **0.9012** | Track 2 (+0.0332) |
| **Macro** | **0.8551** | 0.8343 | **v6 (+0.0208)** |

v6's largest edge is on the rarest class, `threat` (+10.7 F1 points); the transformer's only win is
`neutral`, helped by pretrained language understanding.

---

## 5. Parameter Efficiency

| Model | Params | Macro-F1 | F1 per Million Params | CPU inference |
|-------|--------|----------|-----------------------|---------------|
| v6 Hierarchical | 3.30M | 0.8551 | 0.259 | Yes (<10ms/sample) |
| Track 2 BanglaBERT | ~110M | 0.8343 | 0.0076 | Impractical (~50-100ms) |

v6 is roughly **34x more parameter-efficient** and ships as a ~13MB checkpoint vs ~440MB.

---

## 6. Literature Comparison

F1 across papers is approximate (different datasets, label schemas, protocols); our two models share
identical data and splits. Our work is dated **2026**.

| Year | Work | Model | Params | Dataset | Macro-F1 | CPU |
|------|------|-------|--------|---------|----------|-----|
| 2019 | Emon et al. (ICCIT) | SVM + TF-IDF | <1M | 5.1K | ~0.52 | Yes |
| 2019 | Ishmam & Sharmin (ICCIT) | BiLSTM + Attention | ~3M | 5.1K | ~0.71 | Yes |
| 2020 | Karim et al. (IEEE TCSS) | mBERT (DeepHateExplainer) | ~110M | 44K | ~0.87 | No |
| 2021 | Ahmed et al. (ICCIT) | Ensemble CNN+BiLSTM | ~5M | 10K | ~0.78 | Yes |
| 2022 | Romim et al. (LREC) | BanglaBERT fine-tuned | ~110M | 30K | ~0.84 | No |
| 2023 | Belal et al. (IEEE Access) | CNN-BiLSTM + BanglaBERT | ~115M | 15K | ~0.86 | No |
| 2024 | Saha et al. (Expert Sys. App.) | ToxiFusion (multimodal) | ~150M | 8K | ~0.85 | No |
| **2026** | **Ours — v6 Hierarchical** | **CharCNN+FastText+BiGRU+Attn (2-stage)** | **3.30M** | **12.7K** | **0.8551** | **Yes** |
| 2026 | Ours — Track 2 | BanglaBERT + hierarchical 2-stage head | ~110M | 12.7K | 0.8343 | No |

Observations:
- v6 reaches competitive F1 with **33-45x fewer parameters** than the transformer-based prior work.
- v6 is the only model in this table that exceeds 0.85 Macro-F1 **and** is CPU-deployable.
- Works reporting higher F1 use larger datasets (30-44K) and/or larger/multimodal models.

---

## 7. Running on Kaggle (both notebooks)

1. Accelerator **GPU T4 x2**, **Internet ON** (FastText / BanglaBERT downloads).
2. Attach the dataset; `DATA_PATH` points to the Kaggle path with on-disk fallbacks.
3. Run All. Outputs: `v6_*` (hierarchical) and `transformer_*` (Track 2) checkpoints, summaries,
   curves, and comparison figures.

---

## 8. Where remaining accuracy lives

- **`insult`** (F1 ~0.79) is the hardest class in both models due to heavy overlap with `vulgar`;
  targeted augmentation or a vulgar/insult disambiguation stage is the main lever.
- **Ensemble v6 + Track 2** probabilities: complementary strengths (v6 owns `threat`, Track 2 owns
  `neutral` and ranking/ROC-AUC) — the most promising cross-model gain.
- **Larger data (>50k)** would favor the transformer and is worth re-evaluating if available.
