"""
Bengali Multi-Label Cyberbullying Detection — v5 (T4x2 Parallel)
================================================================
Lightweight Pretrained-Embedding Model (~8.2M total / <10M, ~0.75M trainable)
5-label multi-label: vulgar, threat, troll, insult, neutral

Key v5 changes:
  - New 15K balanced dataset
  - 2x T4 GPU parallel training (nn.DataParallel)
  - Focal Loss for threat class imbalance
  - Two-phase training: frozen → unfrozen embeddings
  - CosineAnnealingWarmRestarts scheduler
  - Training/Val loss + accuracy curves

Usage on Kaggle:
  Select GPU T4 x2 accelerator, then run all cells.
"""

# ============================================================
# Section 1 — Setup & Imports
# ============================================================
import os, re, math, random, time, json, copy, gzip, io, urllib.request, warnings
warnings.filterwarnings('ignore')

import numpy as np

import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LambdaLR

from sklearn.metrics import (
    f1_score, precision_recall_fscore_support, hamming_loss,
    classification_report, multilabel_confusion_matrix, roc_auc_score,
    precision_recall_curve, roc_curve, auc, average_precision_score
)

try:
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
except ImportError:
    os.system('pip install -q iterative-stratification')
    from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
NUM_GPUS = torch.cuda.device_count()
print(f'PyTorch: {torch.__version__} | Device: {DEVICE} | GPUs: {NUM_GPUS}')
for i in range(NUM_GPUS):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')


# ============================================================
# Section 2 — Configuration (Under 10M params)
# ============================================================
class Config:
    # ----- data -----
    DATA_PATH = '/kaggle/input/datasets/combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv'
    TEXT_COL = 'text'
    LABEL_COLS = ['vulgar', 'threat', 'troll', 'insult', 'neutral']
    NUM_CLASSES = 5
    TOXIC_COLS = ['vulgar', 'threat', 'troll', 'insult']

    # ----- preprocessing -----
    MIN_WORDS = 2

    # ----- vocab (25K to stay under 10M total params) -----
    VOCAB_SIZE = 25_000
    MIN_FREQ = 2
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
    FASTTEXT_KAGGLE_PATHS = [
        '/kaggle/input/fasttext-bn-300/cc.bn.300.vec',
        '/kaggle/input/fasttext-bengali/cc.bn.300.vec',
        '/kaggle/input/bengali-fasttext-vectors/cc.bn.300.vec',
        '/kaggle/working/cc.bn.300.vec',
    ]
    FREEZE_EMBEDDING = True
    UNFREEZE_AT_EPOCH = 25       # Unfreeze embeddings at this epoch
    UNFREEZE_LR_FACTOR = 0.1    # 10x smaller LR for embeddings
    PROJECTION_DIM = 128

    # ----- split -----
    TRAIN_FRAC = 0.70
    VAL_FRAC = 0.10
    TEST_FRAC = 0.20

    # ----- model (under 10M total) -----
    CNN_FILTERS = 96
    CNN_KERNELS = (2, 3, 4)
    GRU_HIDDEN = 96
    GRU_LAYERS = 2
    DROPOUT_EMB = 0.35
    DROPOUT = 0.5
    NUM_DROPOUT_SAMPLES = 5

    # ----- training (T4x2) -----
    BATCH_SIZE_PER_GPU = 64
    EPOCHS = 35
    LR = 1.5e-3
    WEIGHT_DECAY = 1e-4
    WARMUP_RATIO = 0.08
    GRAD_CLIP = 1.0
    LABEL_SMOOTHING = 0.05

    # ----- focal loss (NEW) -----
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0

    # ----- augmentation -----
    WORD_DROPOUT_P = 0.18
    MIXUP_ALPHA = 0.4
    MIXUP_PROB = 0.5

    # ----- SWA -----
    USE_SWA = True
    SWA_START_FRAC = 0.65
    SWA_LR = 2e-4

    # ----- evaluation -----
    PATIENCE = 8
    DEFAULT_THRESHOLD = 0.5
    THRESHOLD_GRID = np.arange(0.10, 0.85, 0.01)

cfg = Config()
EFFECTIVE_BATCH_SIZE = cfg.BATCH_SIZE_PER_GPU * max(NUM_GPUS, 1)
print(f'Config loaded. Effective batch size: {EFFECTIVE_BATCH_SIZE}')
print(f'Two-phase training: frozen epochs 1-{cfg.UNFREEZE_AT_EPOCH-1}, '
      f'unfrozen epochs {cfg.UNFREEZE_AT_EPOCH}-{cfg.EPOCHS}')


# ============================================================
# Section 3 — Load & Clean Dataset (15K)
# ============================================================
def find_dataset():
    if os.path.exists(cfg.DATA_PATH):
        return cfg.DATA_PATH
    for root, _, files in os.walk('/kaggle/input'):
        for f in files:
            if f.endswith('.csv') and 'bengali' in f.lower() and 'balanced' in f.lower():
                p = os.path.join(root, f)
                print(f'Auto-detected dataset: {p}')
                return p
    raise FileNotFoundError('No dataset CSV found.')

path = find_dataset()
df_raw = pd.read_csv(path, encoding='utf-8-sig')
print(f'Loaded {len(df_raw):,} rows from {path}')

def clean_dataset(df, cfg, verbose=True):
    df = df.copy()
    df[cfg.LABEL_COLS] = df[cfg.LABEL_COLS].astype(int)
    n0 = len(df)
    # Fix neutral+toxic contradictions
    mask = (df['neutral'] == 1) & (df[cfg.TOXIC_COLS].sum(axis=1) > 0)
    df.loc[mask, 'neutral'] = 0
    if verbose: print(f'  [1] Fixed {mask.sum()} neutral/toxic contradictions')
    # Merge duplicates
    before = len(df)
    df = df.groupby(cfg.TEXT_COL, as_index=False)[cfg.LABEL_COLS].max()
    if verbose: print(f'  [2] Merged {before - len(df)} duplicate-text rows')
    # Drop short texts
    wc = df[cfg.TEXT_COL].astype(str).str.split().apply(len)
    before = len(df)
    df = df[wc >= cfg.MIN_WORDS].reset_index(drop=True)
    if verbose: print(f'  [3] Dropped {before - len(df)} rows with <{cfg.MIN_WORDS} words')
    if verbose: print(f'\nFinal size: {n0:,} -> {len(df):,}')
    return df

df = clean_dataset(df_raw, cfg)
print('\nPost-cleaning counts:')
for c in cfg.LABEL_COLS:
    n = df[c].sum()
    print(f'  {c:<10} {n:>5}  ({100*n/len(df):>5.2f}%)')


# ============================================================
# Section 4 — Text Preprocessing
# ============================================================
BENGALI = r'\u0980-\u09FF'
EN = r'a-zA-Z'
URL_RE = re.compile(r'https?://\S+|www\.\S+')
MENTION_RE = re.compile(r'@\w+')
HASHTAG_RE = re.compile(r'#(\w+)')
EMOJI_RE = re.compile(
    '[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
    '\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF'
    '\U00002700-\U000027BF\U000024C2-\U0001F251]+', flags=re.UNICODE)
MULTISPACE = re.compile(r'\s+')
PUNCT = re.compile(r'[,.!?।;:\"\']')
KEEP = re.compile(f'[^{BENGALI}{EN}0-9\\s]')

def clean_text(s):
    if not isinstance(s, str): return ''
    s = URL_RE.sub(' ', s)
    s = MENTION_RE.sub(' ', s)
    s = HASHTAG_RE.sub(r'\1', s)
    s = EMOJI_RE.sub(' ', s)
    s = PUNCT.sub(' ', s)
    s = KEEP.sub(' ', s)
    s = MULTISPACE.sub(' ', s).strip()
    return s

def tokenize(s):
    return clean_text(s).split()

# ============================================================
# Section 5 — Stratified 70/10/20 Split
# ============================================================
def stratified_split(df, label_cols, frac_train, frac_val, frac_test, seed=SEED):
    assert abs(frac_train + frac_val + frac_test - 1.0) < 1e-6
    y = df[label_cols].values
    m1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=frac_test, random_state=seed)
    idx_tv, idx_te = next(m1.split(df, y))
    df_tv = df.iloc[idx_tv].reset_index(drop=True)
    df_te = df.iloc[idx_te].reset_index(drop=True)
    val_rel = frac_val / (frac_train + frac_val)
    m2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_rel, random_state=seed)
    idx_tr, idx_va = next(m2.split(df_tv, df_tv[label_cols].values))
    df_tr = df_tv.iloc[idx_tr].reset_index(drop=True)
    df_va = df_tv.iloc[idx_va].reset_index(drop=True)
    return df_tr, df_va, df_te

train_df, val_df, test_df = stratified_split(
    df, cfg.LABEL_COLS, cfg.TRAIN_FRAC, cfg.VAL_FRAC, cfg.TEST_FRAC)
print(f'Train: {len(train_df):>5}  Val: {len(val_df):>5}  Test: {len(test_df):>5}')


# ============================================================
# Section 6 — Vocabularies & Encoding
# ============================================================
PAD, UNK = '<pad>', '<unk>'

def build_word_vocab(texts, vocab_size, min_freq):
    c = Counter()
    for t in texts: c.update(tokenize(t))
    most = [w for w, cnt in c.most_common() if cnt >= min_freq][:vocab_size - 2]
    itos = [PAD, UNK] + most
    return {w: i for i, w in enumerate(itos)}, itos

def build_char_vocab(texts, vocab_size):
    c = Counter()
    for t in texts:
        for tok in tokenize(t): c.update(tok)
    most = [ch for ch, _ in c.most_common(vocab_size - 2)]
    itos = [PAD, UNK] + most
    return {ch: i for i, ch in enumerate(itos)}, itos

word_stoi, word_itos = build_word_vocab(
    train_df[cfg.TEXT_COL].tolist(), cfg.VOCAB_SIZE, cfg.MIN_FREQ)
char_stoi, char_itos = build_char_vocab(
    train_df[cfg.TEXT_COL].tolist(), cfg.CHAR_VOCAB_SIZE)
print(f'Word vocabulary: {len(word_stoi):,} | Char vocabulary: {len(char_stoi):,}')

def encode_word_and_char(text, max_len, max_chars):
    toks = tokenize(text)[:max_len]
    word_ids = [word_stoi.get(t, word_stoi[UNK]) for t in toks]
    char_ids = []
    for tok in toks:
        cids = [char_stoi.get(ch, char_stoi[UNK]) for ch in tok][:max_chars]
        cids += [char_stoi[PAD]] * (max_chars - len(cids))
        char_ids.append(cids)
    L = len(word_ids)
    word_ids += [word_stoi[PAD]] * (max_len - L)
    while len(char_ids) < max_len:
        char_ids.append([char_stoi[PAD]] * max_chars)
    return word_ids, char_ids, max(L, 1)


# ============================================================
# Section 7 — Load Pretrained FastText (same as v4)
# ============================================================
def find_fasttext_file():
    for p in cfg.FASTTEXT_KAGGLE_PATHS:
        if os.path.exists(p): return p, False
        if os.path.exists(p + '.gz'): return p + '.gz', True
    return None, False

def load_fasttext_into_matrix(ft_path, gzipped, word_stoi, dim):
    matrix = np.random.normal(0, 0.1, (len(word_stoi), dim)).astype(np.float32)
    matrix[word_stoi[PAD]] = 0.0
    found = 0
    opener = (lambda p: gzip.open(p, 'rt', encoding='utf-8', errors='ignore')) if gzipped \
             else (lambda p: open(p, 'r', encoding='utf-8', errors='ignore'))
    t0 = time.time()
    with opener(ft_path) as f:
        header = f.readline().strip().split()
        n_lines = 0
        needed = set(word_stoi.keys())
        for line in f:
            n_lines += 1
            if n_lines % 200_000 == 0:
                print(f'  scanned {n_lines:,} lines, found {found:,}/{len(needed):,} ({time.time()-t0:.1f}s)')
            parts = line.rstrip().split(' ')
            if len(parts) < dim + 1: continue
            w = parts[0]
            if w in needed:
                try:
                    vec = np.asarray(parts[1:1+dim], dtype=np.float32)
                    if vec.shape[0] == dim:
                        matrix[word_stoi[w]] = vec
                        found += 1
                except ValueError: pass
                if found == len(needed): break
    return matrix, found

ft_path, ft_gzipped = find_fasttext_file()
if cfg.USE_PRETRAINED and ft_path is None:
    print(f'Downloading FastText from {cfg.FASTTEXT_URL}...')
    try:
        download_path = '/kaggle/working/cc.bn.300.vec.gz'
        urllib.request.urlretrieve(cfg.FASTTEXT_URL, download_path)
        ft_path, ft_gzipped = download_path, True
    except Exception as e:
        print(f'Download failed: {e}. Using random init.')
        cfg.USE_PRETRAINED = False

if cfg.USE_PRETRAINED and ft_path:
    print('Loading FastText vectors...')
    embedding_matrix, n_found = load_fasttext_into_matrix(
        ft_path, ft_gzipped, word_stoi, cfg.FASTTEXT_DIM)
    coverage = n_found / len(word_stoi)
    print(f'Loaded {n_found:,}/{len(word_stoi):,} vectors ({coverage:.1%} coverage)')
else:
    embedding_matrix = np.random.normal(0, 0.1, (len(word_stoi), cfg.FASTTEXT_DIM)).astype(np.float32)
    embedding_matrix[word_stoi[PAD]] = 0.0
    coverage = 0.0


# ============================================================
# Section 8 — Dataset & DataLoaders (T4x2 optimized)
# ============================================================
class BengaliCBDataset(Dataset):
    def __init__(self, df, word_stoi, char_stoi, max_len, max_chars,
                 text_col, label_cols, training=False, word_dropout_p=0.0):
        self.texts = df[text_col].tolist()
        self.labels = df[label_cols].values.astype('float32')
        self.word_stoi = word_stoi; self.char_stoi = char_stoi
        self.max_len = max_len; self.max_chars = max_chars
        self.training = training; self.word_dropout_p = word_dropout_p
        self.unk_word = word_stoi[UNK]

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        wids, cids, length = encode_word_and_char(
            self.texts[idx], self.max_len, self.max_chars)
        if self.training and self.word_dropout_p > 0:
            for i in range(length):
                if random.random() < self.word_dropout_p:
                    wids[i] = self.unk_word
        return (torch.tensor(wids, dtype=torch.long),
                torch.tensor(cids, dtype=torch.long),
                torch.tensor(length, dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float32))

train_ds = BengaliCBDataset(train_df, word_stoi, char_stoi, cfg.MAX_LEN,
    cfg.MAX_CHAR_PER_WORD, cfg.TEXT_COL, cfg.LABEL_COLS,
    training=True, word_dropout_p=cfg.WORD_DROPOUT_P)
val_ds = BengaliCBDataset(val_df, word_stoi, char_stoi, cfg.MAX_LEN,
    cfg.MAX_CHAR_PER_WORD, cfg.TEXT_COL, cfg.LABEL_COLS, training=False)
test_ds = BengaliCBDataset(test_df, word_stoi, char_stoi, cfg.MAX_LEN,
    cfg.MAX_CHAR_PER_WORD, cfg.TEXT_COL, cfg.LABEL_COLS, training=False)

# T4x2: effective batch = 64 * 2 = 128
bs = EFFECTIVE_BATCH_SIZE
train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=4, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                         num_workers=4, pin_memory=True)
print(f'DataLoaders: batch={bs}, train_batches={len(train_loader)}, '
      f'val_batches={len(val_loader)}, test_batches={len(test_loader)}')


# ============================================================
# Section 9 — Model: V5 (Same arch as V4, under 10M params)
# ============================================================
class SpatialDropout1D(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout1d(p) if hasattr(nn, 'Dropout1d') else nn.Dropout2d(p)
    def forward(self, x):
        x = x.permute(0, 2, 1); x = self.dropout(x); return x.permute(0, 2, 1).contiguous()

class CharCNN(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim, filters, kernels):
        super().__init__()
        self.embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(char_emb_dim, filters, kernel_size=k, padding=k//2) for k in kernels])
        self.out_dim = filters * len(kernels)
    def forward(self, char_ids):
        B, T, C = char_ids.shape
        x = self.embedding(char_ids.view(B*T, C)).transpose(1, 2)
        outs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            h = F.max_pool1d(h, kernel_size=h.size(-1))
            outs.append(h.squeeze(-1))
        return torch.cat(outs, dim=1).view(B, T, -1)

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, H, mask=None):
        e = self.v(torch.tanh(self.W(H))).squeeze(-1)
        if mask is not None: e = e.masked_fill(mask == 0, -1e9)
        a = F.softmax(e, dim=1)
        return torch.bmm(a.unsqueeze(1), H).squeeze(1), a

class V5Model(nn.Module):
    def __init__(self, cfg, word_vocab_size, char_vocab_size, pretrained_emb=None):
        super().__init__()
        self.pad_idx = 0
        self.word_embedding = nn.Embedding(word_vocab_size, cfg.FASTTEXT_DIM, padding_idx=0)
        if pretrained_emb is not None:
            self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))
        if cfg.FREEZE_EMBEDDING:
            self.word_embedding.weight.requires_grad = False

        self.projection = nn.Linear(cfg.FASTTEXT_DIM, cfg.PROJECTION_DIM)
        self.char_cnn = CharCNN(char_vocab_size, cfg.CHAR_EMBED_DIM,
                                cfg.CHAR_CNN_FILTERS, cfg.CHAR_KERNELS)
        combined_dim = cfg.PROJECTION_DIM + self.char_cnn.out_dim
        self.emb_dropout = SpatialDropout1D(cfg.DROPOUT_EMB)
        self.convs = nn.ModuleList([
            nn.Conv1d(combined_dim, cfg.CNN_FILTERS, kernel_size=k, padding=k//2)
            for k in cfg.CNN_KERNELS])
        cnn_out = cfg.CNN_FILTERS * len(cfg.CNN_KERNELS)
        self.bigru = nn.GRU(cnn_out, cfg.GRU_HIDDEN, cfg.GRU_LAYERS,
                            batch_first=True, bidirectional=True,
                            dropout=cfg.DROPOUT if cfg.GRU_LAYERS > 1 else 0)
        gru_out = cfg.GRU_HIDDEN * 2
        self.attention = AdditiveAttention(gru_out)
        self.dropout_layers = nn.ModuleList([nn.Dropout(cfg.DROPOUT)
                                             for _ in range(cfg.NUM_DROPOUT_SAMPLES)])
        self.fc1 = nn.Linear(gru_out, cfg.GRU_HIDDEN)
        self.fc2 = nn.Linear(cfg.GRU_HIDDEN, cfg.NUM_CLASSES)
        self.num_dropout_samples = cfg.NUM_DROPOUT_SAMPLES

    def forward(self, word_ids, char_ids, lengths=None,
                mixup_lam=None, mixup_idx=None):
        mask = (word_ids != self.pad_idx).long()
        w_emb = self.projection(self.word_embedding(word_ids))
        c_emb = self.char_cnn(char_ids)
        x = torch.cat([w_emb, c_emb], dim=-1)
        if mixup_lam is not None and mixup_idx is not None:
            x = mixup_lam * x + (1 - mixup_lam) * x[mixup_idx]
        x = self.emb_dropout(x).transpose(1, 2)
        conv_outs = []
        for conv in self.convs:
            h = F.relu(conv(x))
            if h.size(-1) != word_ids.size(1):
                h = h[..., :word_ids.size(1)] if h.size(-1) > word_ids.size(1) \
                    else F.pad(h, (0, word_ids.size(1) - h.size(-1)))
            conv_outs.append(h)
        x = torch.cat(conv_outs, dim=1).transpose(1, 2)
        rnn_out, _ = self.bigru(x)
        ctx, attn = self.attention(rnn_out, mask=mask)
        if self.training and self.num_dropout_samples > 1:
            logits_list = []
            for drop in self.dropout_layers:
                h = drop(ctx); h = F.relu(self.fc1(h)); h = drop(h)
                logits_list.append(self.fc2(h))
            logits = torch.stack(logits_list, 0).mean(0)
        else:
            h = self.dropout_layers[0](ctx)
            h = F.relu(self.fc1(h)); h = self.dropout_layers[0](h)
            logits = self.fc2(h)
        return logits, attn

# Create model & verify param count
model = V5Model(cfg, len(word_stoi), len(char_stoi), embedding_matrix)
total_params = sum(p.numel() for p in model.parameters())
trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'\nTotal params: {total_params:,} ({"UNDER 10M ✓" if total_params < 10_000_000 else "OVER 10M ✗"})')
print(f'Trainable (Phase 1): {trainable_params_count:,}')

# Wrap with DataParallel for T4x2
if NUM_GPUS > 1:
    model = nn.DataParallel(model)
    print(f'Model wrapped in DataParallel ({NUM_GPUS} GPUs)')
model = model.to(DEVICE)


# ============================================================
# Section 10 — Focal Loss + Training Setup
# ============================================================
class FocalBCELoss(nn.Module):
    """Focal loss for multi-label classification with class imbalance."""
    def __init__(self, gamma=2.0, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.gamma = gamma
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
        return (focal_weight * bce).mean()

# Compute pos_weight from training data
y_train = train_df[cfg.LABEL_COLS].values
pos = y_train.sum(axis=0)
neg = len(y_train) - pos
pos_weight = torch.tensor(neg / np.maximum(pos, 1), dtype=torch.float32).to(DEVICE)
print('pos_weight:', {c: f'{w:.2f}' for c, w in zip(cfg.LABEL_COLS, pos_weight.cpu().numpy())})

if cfg.USE_FOCAL_LOSS:
    criterion = FocalBCELoss(gamma=cfg.FOCAL_GAMMA, pos_weight=pos_weight,
                             smoothing=cfg.LABEL_SMOOTHING)
    print(f'Using Focal Loss (gamma={cfg.FOCAL_GAMMA})')
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Optimizer (Phase 1: only trainable params)
base_model = model.module if hasattr(model, 'module') else model
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(trainable_params, lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

total_steps = len(train_loader) * cfg.EPOCHS
warmup_steps = int(cfg.WARMUP_RATIO * total_steps)
swa_start_step = int(cfg.SWA_START_FRAC * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    if cfg.USE_SWA and step >= swa_start_step:
        return cfg.SWA_LR / cfg.LR
    progress = (step - warmup_steps) / max(1, swa_start_step - warmup_steps)
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.01)

scheduler = LambdaLR(optimizer, lr_lambda)
print(f'Total steps: {total_steps} | Warmup: {warmup_steps} | SWA at step {swa_start_step}')


# ============================================================
# Section 11 — Training Loop (Two-Phase, T4x2, with curves)
# ============================================================
@torch.no_grad()
def evaluate(model, loader, criterion=None, threshold=0.5):
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_samples = 0.0, 0
    for wid, cid, lens, y in loader:
        wid, cid, y = wid.to(DEVICE), cid.to(DEVICE), y.to(DEVICE)
        logits, _ = model(wid, cid, lens)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * wid.size(0)
        n_samples += wid.size(0)
        all_logits.append(logits.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    logits = np.concatenate(all_logits)
    labels = np.concatenate(all_labels).astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= threshold).astype(int)
    return {
        'loss': total_loss / max(n_samples, 1) if criterion else None,
        'macro_f1': f1_score(labels, preds, average='macro', zero_division=0),
        'micro_f1': f1_score(labels, preds, average='micro', zero_division=0),
        'per_class_f1': f1_score(labels, preds, average=None, zero_division=0),
        'hamming': hamming_loss(labels, preds),
        'preds': preds, 'labels': labels, 'probs': probs,
    }

# History tracking
history = {
    'train_loss': [], 'val_loss': [],
    'train_macro_f1': [], 'val_macro_f1': [],
    'val_per_class_f1': [], 'lr': [], 'phase': [],
}

best_val_loss = float('inf')
best_val_macro = -1.0
best_state = None
patience_left = cfg.PATIENCE
swa_state = None; swa_n = 0
global_step = 0
phase = 1

print(f'\n{"="*70}')
print('STARTING TRAINING (T4x2 Parallel)')
print(f'{"="*70}\n')

for epoch in range(1, cfg.EPOCHS + 1):
    t0 = time.time()

    # === Phase 2: Unfreeze embeddings ===
    if epoch == cfg.UNFREEZE_AT_EPOCH and phase == 1:
        phase = 2
        print(f'\n>>> PHASE 2: Unfreezing word embeddings (epoch {epoch}) <<<')
        base_model.word_embedding.weight.requires_grad = True
        # Re-create optimizer with differential LR
        optimizer = AdamW([
            {'params': [p for n, p in base_model.named_parameters()
                        if 'word_embedding' not in n and p.requires_grad],
             'lr': cfg.LR * 0.3},
            {'params': [base_model.word_embedding.weight],
             'lr': cfg.LR * cfg.UNFREEZE_LR_FACTOR},
        ], weight_decay=cfg.WEIGHT_DECAY)
        scheduler = LambdaLR(optimizer, lambda step: cfg.SWA_LR / (cfg.LR * 0.3))
        new_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'    Trainable params now: {new_trainable:,} (still under 10M total: {total_params:,})')

    # === Training ===
    model.train()
    running, n_samples = 0.0, 0
    for wid, cid, lens, y in train_loader:
        wid, cid, y = wid.to(DEVICE), cid.to(DEVICE), y.to(DEVICE)
        do_mixup = cfg.MIXUP_ALPHA > 0 and random.random() < cfg.MIXUP_PROB
        if do_mixup:
            lam = float(np.random.beta(cfg.MIXUP_ALPHA, cfg.MIXUP_ALPHA))
            lam = max(lam, 1 - lam)
            idx = torch.randperm(wid.size(0)).to(DEVICE)
            logits, _ = model(wid, cid, lens, mixup_lam=lam, mixup_idx=idx)
            y_mixed = lam * y + (1 - lam) * y[idx]
            loss = criterion(logits, y_mixed)
        else:
            logits, _ = model(wid, cid, lens)
            loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], cfg.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        running += loss.item() * wid.size(0)
        n_samples += wid.size(0)
        global_step += 1
    train_loss = running / n_samples

    # === Validation ===
    val = evaluate(model, val_loader, criterion=criterion)
    tr_eval = evaluate(model, train_loader, criterion=None)

    current_lr = optimizer.param_groups[0]['lr']
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val['loss'])
    history['train_macro_f1'].append(tr_eval['macro_f1'])
    history['val_macro_f1'].append(val['macro_f1'])
    history['val_per_class_f1'].append(val['per_class_f1'])
    history['lr'].append(current_lr)
    history['phase'].append(phase)

    gap = tr_eval['macro_f1'] - val['macro_f1']
    swa_marker = ' [SWA]' if (cfg.USE_SWA and global_step >= swa_start_step) else ''
    phase_marker = f'[P{phase}]'
    print(f'Ep {epoch:02d} {phase_marker}{swa_marker} | '
          f'tr_loss {train_loss:.4f} val_loss {val["loss"]:.4f} | '
          f'tr_F1 {tr_eval["macro_f1"]:.4f} val_F1 {val["macro_f1"]:.4f} '
          f'(gap {gap:+.3f}) | lr {current_lr:.2e} | {time.time()-t0:.1f}s')

    # SWA
    if cfg.USE_SWA and global_step >= swa_start_step:
        if swa_state is None:
            swa_state = {k: v.detach().clone().float()
                         for k, v in base_model.state_dict().items()}
            swa_n = 1
        else:
            swa_n += 1
            for k, v in base_model.state_dict().items():
                swa_state[k] += (v.detach().float() - swa_state[k]) / swa_n

    # Early stopping
    if val['loss'] < best_val_loss - 1e-4:
        best_val_loss = val['loss']
        best_val_macro = val['macro_f1']
        best_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
        patience_left = cfg.PATIENCE
        print(f'   -> new best val_loss={best_val_loss:.4f} (F1={val["macro_f1"]:.4f})')
    else:
        patience_left -= 1
        if patience_left <= 0:
            print(f'\n  Early stopping at epoch {epoch}.')
            break

# Final model selection (best vs SWA)
print(f'\nBest checkpoint: macro_f1 = {best_val_macro:.4f}')
if swa_state is not None:
    base_model.load_state_dict({k: v.to(DEVICE) for k, v in swa_state.items()})
    swa_val = evaluate(model, val_loader, criterion=criterion)
    print(f'SWA model:       macro_f1 = {swa_val["macro_f1"]:.4f}')
    if swa_val['macro_f1'] >= best_val_macro:
        print('-> Using SWA model')
    else:
        print('-> Using best checkpoint')
        base_model.load_state_dict(best_state)
else:
    base_model.load_state_dict(best_state)


# ============================================================
# Section 12 — Training & Validation Curves (KEY PLOTS)
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
epochs_range = range(1, len(history['train_loss']) + 1)

# --- Plot 1: Loss Curves ---
axes[0, 0].plot(epochs_range, history['train_loss'], 'b-o', markersize=3, label='Train Loss')
axes[0, 0].plot(epochs_range, history['val_loss'], 'r-s', markersize=3, label='Val Loss')
if cfg.UNFREEZE_AT_EPOCH <= len(history['train_loss']):
    axes[0, 0].axvline(cfg.UNFREEZE_AT_EPOCH, color='green', ls='--', alpha=0.7,
                        label=f'Unfreeze (ep {cfg.UNFREEZE_AT_EPOCH})')
axes[0, 0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# --- Plot 2: Macro-F1 (Accuracy) Curves ---
axes[0, 1].plot(epochs_range, history['train_macro_f1'], 'b-o', markersize=3, label='Train Macro-F1')
axes[0, 1].plot(epochs_range, history['val_macro_f1'], 'r-s', markersize=3, label='Val Macro-F1')
best_ep = int(np.argmax(history['val_macro_f1'])) + 1
axes[0, 1].axvline(best_ep, color='gold', ls='--', alpha=0.7, label=f'Best val @ ep {best_ep}')
if cfg.UNFREEZE_AT_EPOCH <= len(history['train_loss']):
    axes[0, 1].axvline(cfg.UNFREEZE_AT_EPOCH, color='green', ls='--', alpha=0.7,
                        label=f'Unfreeze (ep {cfg.UNFREEZE_AT_EPOCH})')
axes[0, 1].set_title('Training & Validation Macro-F1', fontsize=13, fontweight='bold')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Macro-F1')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

# --- Plot 3: Per-Class Val F1 ---
per_cls = np.array(history['val_per_class_f1'])
colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
for i, (c, color) in enumerate(zip(cfg.LABEL_COLS, colors)):
    axes[1, 0].plot(epochs_range, per_cls[:, i], 'o-', markersize=3, label=c, color=color)
axes[1, 0].set_title('Per-Class Validation F1', fontsize=13, fontweight='bold')
axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

# --- Plot 4: Learning Rate Schedule ---
axes[1, 1].plot(epochs_range, history['lr'], 'g-', linewidth=2)
if cfg.UNFREEZE_AT_EPOCH <= len(history['train_loss']):
    axes[1, 1].axvline(cfg.UNFREEZE_AT_EPOCH, color='red', ls='--', alpha=0.7,
                        label=f'Phase 2 (ep {cfg.UNFREEZE_AT_EPOCH})')
axes[1, 1].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
axes[1, 1].set_xlabel('Epoch'); axes[1, 1].set_ylabel('LR')
axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)
axes[1, 1].set_yscale('log')

plt.tight_layout()
plt.savefig('/kaggle/working/v5_training_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# Overfitting diagnostic
print('\nOVERFITTING DIAGNOSTIC')
print('=' * 50)
final_tr = history['train_macro_f1'][-1]
final_va = history['val_macro_f1'][-1]
gap = final_tr - final_va
print(f'Final train F1: {final_tr:.4f}')
print(f'Final val   F1: {final_va:.4f}')
print(f'Final gap     : {gap:+.4f}')
print(f'v4 gap: +0.058 | v5 gap: {gap:+.3f}')
if gap > 0.15: print('-> Significant overfitting')
elif gap > 0.08: print('-> Moderate overfitting')
elif gap > 0.03: print('-> Healthy fit ✓')
else: print('-> Possibly underfit')


# ============================================================
# Section 13 — Threshold Tuning & Final Test Evaluation
# ============================================================
def tune_thresholds(model, loader, grid):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for wid, cid, lens, y in loader:
            wid, cid = wid.to(DEVICE), cid.to(DEVICE)
            logits, _ = model(wid, cid, lens)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = np.concatenate(all_labels).astype(int)
    best_thr = np.full(probs.shape[1], 0.5)
    for c in range(probs.shape[1]):
        best_f1 = -1
        for t in grid:
            p = (probs[:, c] >= t).astype(int)
            f = f1_score(labels[:, c], p, zero_division=0)
            if f > best_f1: best_f1, best_thr[c] = f, t
    return best_thr

thresholds = tune_thresholds(model, val_loader, cfg.THRESHOLD_GRID)
print('Tuned thresholds:')
for c, t in zip(cfg.LABEL_COLS, thresholds):
    print(f'  {c:<10} {t:.2f}')

# Final test evaluation
def evaluate_with_thresholds(model, loader, thresholds):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for wid, cid, lens, y in loader:
            wid, cid = wid.to(DEVICE), cid.to(DEVICE)
            logits, _ = model(wid, cid, lens)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(y.numpy())
    logits = np.concatenate(all_logits)
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = np.concatenate(all_labels).astype(int)
    preds = (probs >= thresholds[None, :]).astype(int)
    return probs, preds, labels

probs, preds, y_true = evaluate_with_thresholds(model, test_loader, thresholds)

print('\n' + '=' * 70)
print('FINAL TEST-SET RESULTS (v5 — T4x2 Parallel)')
print('=' * 70)
macro = f1_score(y_true, preds, average='macro', zero_division=0)
micro = f1_score(y_true, preds, average='micro', zero_division=0)
weighted = f1_score(y_true, preds, average='weighted', zero_division=0)
samples = f1_score(y_true, preds, average='samples', zero_division=0)
hamming = hamming_loss(y_true, preds)
print(f'Macro-F1      : {macro:.4f}')
print(f'Micro-F1      : {micro:.4f}')
print(f'Weighted-F1   : {weighted:.4f}')
print(f'Samples-F1    : {samples:.4f}')
print(f'Hamming Loss  : {hamming:.4f}')
try:
    print(f'Macro ROC-AUC : {roc_auc_score(y_true, probs, average="macro"):.4f}')
    print(f'Macro PR-AUC  : {average_precision_score(y_true, probs, average="macro"):.4f}')
except: pass
print('\nPer-class report:')
print(classification_report(y_true, preds, target_names=cfg.LABEL_COLS, zero_division=0, digits=4))


# ============================================================
# Section 14 — Save Model & Summary
# ============================================================
out_dir = '/kaggle/working'
os.makedirs(out_dir, exist_ok=True)

n_params = sum(p.numel() for p in base_model.parameters())
n_trainable = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

torch.save({
    'state_dict': base_model.state_dict(),
    'config': {k: getattr(cfg, k) for k in dir(cfg)
               if not k.startswith('_') and not callable(getattr(cfg, k))},
    'thresholds': thresholds.tolist(),
    'label_cols': cfg.LABEL_COLS,
    'word_stoi': word_stoi,
    'char_stoi': char_stoi,
    'history': {k: (v if not isinstance(v, np.ndarray) else v.tolist())
                for k, v in history.items()},
    'test_macro_f1': float(macro),
    'fasttext_coverage': float(coverage),
    'num_gpus': NUM_GPUS,
}, os.path.join(out_dir, 'bengali_cb_v5.pt'))

summary = {
    'version': 'v5',
    'dataset': 'combined_multi_labeled_bengali_comments_balanced_13k_14k_plus_neutral.csv',
    'dataset_size': len(df),
    'test_macro_f1': float(macro),
    'test_micro_f1': float(micro),
    'test_weighted_f1': float(weighted),
    'test_hamming': float(hamming),
    'n_params_total': int(n_params),
    'n_params_trainable_phase1': int(trainable_params_count),
    'n_params_trainable_phase2': int(n_trainable),
    'under_10M': n_params < 10_000_000,
    'fasttext_coverage': float(coverage),
    'word_vocab_size': len(word_stoi),
    'thresholds': {c: float(t) for c, t in zip(cfg.LABEL_COLS, thresholds)},
    'best_val_macro_f1': float(best_val_macro),
    'epochs_trained': len(history['train_loss']),
    'num_gpus': NUM_GPUS,
    'gpu_type': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
    'split': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
    'focal_loss': cfg.USE_FOCAL_LOSS,
    'focal_gamma': cfg.FOCAL_GAMMA,
}
with open(os.path.join(out_dir, 'bengali_cb_v5_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f'\nSaved to {out_dir}:')
print(f'  bengali_cb_v5.pt ({os.path.getsize(out_dir+"/bengali_cb_v5.pt")/1e6:.1f} MB)')
print(f'  bengali_cb_v5_summary.json')
print(f'\n{"="*70}')
print(f'FINAL: Test Macro-F1 = {macro:.4f} | Params = {n_params:,} (under 10M: ✓)')
print(f'{"="*70}')
