import os, json, pickle, re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import f1_score, classification_report

# ── KONFIGURASI ───────────────────────────────────────────────────────────────
CONFIG = {
    'model_name'   : 'indobenchmark/indobert-base-p1',
    'max_length'   : 128,
    'dropout_rate' : 0.17366466880936898,
    'batch_size'   : 8,
    'learning_rate': 4.801650729207848e-05,
    'num_epochs'   : 25,
    'warmup_ratio' : 0.05366045312634687,
    'weight_decay' : 0.039008177788135416,
    'max_grad_norm': 1.0,
    'patience'     : 5,
    'save_dir'     : './model_output_v2',
}

ASPECTS = [
    'Content Quality',
    'Subscription & Pricing',
    'UI/UX',
    'Functionality',
    'Technical & Access',
]

# Skema: 0=Neg | 1=Neu | 2=Pos | 3=None
NUM_CLASSES = {
    'Content Quality'       : 4,
    'Subscription & Pricing': 4,
    'UI/UX'                 : 4,
    'Functionality'         : 4,
    'Technical & Access'    : 3,
}

LABEL_NAMES = {
    'Content Quality'       : ['Negatif', 'Netral', 'Positif', 'None'],
    'Subscription & Pricing': ['Negatif', 'Netral', 'Positif', 'None'],
    'UI/UX'                 : ['Negatif', 'Netral', 'Positif', 'None'],
    'Functionality'         : ['Negatif', 'Netral', 'Positif', 'None'],
    'Technical & Access'    : ['Negatif', 'Positif', 'None'],
}

# None index per aspek (indeks kelas None)
NONE_IDX = {asp: NUM_CLASSES[asp] - 1 for asp in ASPECTS}

# Class weights dari training set — urutan: Neg / Neu / Pos / None
# (untuk Technical & Access: Neg / Pos / None)
# Perbarui nilai ini dari class_weights.json hasil data_preperation.py
# Class weights — None dikecilkan ke 0.10 agar model lebih fokus ke sentimen
CLASS_WEIGHTS = {
      "Content Quality": [
    7.165,
    12.35344827586207,
    1.1482371794871795,
    0.34380998080614206
  ],
  "Subscription & Pricing": [
    5.045774647887324,
    18.855263157894736,
    21.073529411764707,
    0.2701734539969834
  ],
  "UI/UX": [
    8.331395348837209,
    35.825,
    16.28409090909091,
    0.26380706921944036
  ],
  "Functionality": [
    6.513636363636364,
    17.05952380952381,
    32.56818181818182,
    0.2661589895988113
  ],
  "Technical & Access": [
    3.9153005464480874,
    25.140350877192983,
    0.3697110423116615
  ]
}

SLANG_DICT = {
    'gak':'tidak','ga':'tidak','nggak':'tidak','ngga':'tidak','gk':'tidak',
    'tdk':'tidak','yg':'yang','dgn':'dengan','dg':'dengan','utk':'untuk',
    'krn':'karena','karna':'karena','klo':'kalau','kl':'kalau',
    'udah':'sudah','udh':'sudah','sdh':'sudah','blm':'belum',
    'lg':'lagi','lgs':'langsung','sy':'saya','bgt':'banget',
    'bs':'bisa','bsa':'bisa','tp':'tapi','ttp':'tetap','jd':'jadi',
    'app':'aplikasi','apps':'aplikasi','apk':'aplikasi',
    'ok':'oke','subs':'berlangganan',
    'langgan':'berlangganan','langgannya':'berlangganannya',
    'langganan':'berlangganan','berlangganannya':'berlangganannya',
    'harga':'harga','mahal':'mahal',
    'epaper':'e-paper','e paper':'e-paper',
}

def preprocess_text(text: str) -> str:
    """
    Preprocessing minimal untuk IndoBERT.
    Tidak perlu stemming/stopword — BERT sudah contextual.

    Langkah:
      1. Hapus emoji (non-ASCII)
      2. Lowercase
      3. Hapus URL dan mention
      4. Compress karakter berulang >2 (mahaaal → mahaal)
      5. Hapus karakter tidak perlu, pertahankan ?.!.,
      6. Normalisasi spasi
      7. Normalisasi kata slang (word-level)
    """
    if pd.isna(text) or str(text).strip() == '':
        return ''
    text = str(text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|@\w+', '', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r"[^a-z0-9\s\?\!\.\,\-']", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [SLANG_DICT.get(w, w) for w in text.split()]
    text  = ' '.join(words)
    text  = re.sub(r'\s+', ' ', text).strip()
    return text

# ── DATASET ───────────────────────────────────────────────────────────────────
class ABSADataset(Dataset):
    def __init__(self, df, tokenizer, max_length, aspects):
        self.df        = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length= max_length
        self.aspects   = aspects
        self.label_cols= [f'lbl_{a}' for a in aspects]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row['komentar_clean'])

        encoding = self.tokenizer(text, max_length=self.max_length, truncation=True)

        labels = {}
        for col, aspect in zip(self.label_cols, self.aspects):
            labels[aspect] = torch.tensor(int(row[col]), dtype=torch.long)

        return {
            'input_ids'     : encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels'        : labels,
        }


class ABSACollator:
    def __init__(self, tokenizer):
        self._base = DataCollatorWithPadding(tokenizer, return_tensors='pt')

    def __call__(self, features):
        text_feats  = [{'input_ids': f['input_ids'],
                        'attention_mask': f['attention_mask']}
                       for f in features]
        labels_list = [f['labels'] for f in features]
        batch = self._base(text_feats)
        batch['labels'] = {
            asp: torch.stack([l[asp] for l in labels_list])
            for asp in labels_list[0]
        }
        return batch


# ── MODEL ─────────────────────────────────────────────────────────────────────
class ABSAModel(nn.Module):
    def __init__(self, model_name, aspects, num_classes, dropout_rate):
        super().__init__()
        self.encoder  = AutoModel.from_pretrained(model_name)
        hidden        = self.encoder.config.hidden_size   # 768
        self.dropout  = nn.Dropout(dropout_rate)
        self.heads    = nn.ModuleDict({
            asp.replace(' ','_').replace('&','and'):
                nn.Linear(hidden, num_classes[asp])
            for asp in aspects
        })
        self.aspects = aspects

    def _key(self, asp):
        return asp.replace(' ','_').replace('&','and')

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0, :])
        return {asp: self.heads[self._key(asp)](cls) for asp in self.aspects}


# ── LOSS ──────────────────────────────────────────────────────────────────────
def compute_loss(logits_dict, labels_dict, class_weights_dict, device):
    total = torch.tensor(0.0, device=device)
    for asp, logits in logits_dict.items():
        labels  = labels_dict[asp].to(device)
        weights = torch.tensor(class_weights_dict[asp],
                               dtype=torch.float32, device=device)
        total  += F.cross_entropy(logits, labels, weight=weights)
    return total / len(logits_dict)


# ── EVALUASI ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    """
    Mengembalikan DUA metrik:
      Detection F1  : macro F1 semua kelas termasuk None
                      → mengukur kemampuan deteksi aspek
      Sentiment F1  : macro F1 kelas sentimen saja (exclude None)
                      → mengukur kualitas klasifikasi sentimen
    """
    model.eval()
    all_preds  = {a: [] for a in ASPECTS}
    all_labels = {a: [] for a in ASPECTS}

    with torch.no_grad():
        for batch in loader:
            iids  = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            logits= model(iids, amask)
            for asp in ASPECTS:
                preds  = logits[asp].argmax(dim=-1).cpu().numpy()
                labels = batch['labels'][asp].numpy()
                all_preds[asp].extend(preds.tolist())
                all_labels[asp].extend(labels.tolist())

    detect_f1  = {}   # include None
    sentiment_f1 = {} # exclude None

    for asp in ASPECTS:
        y_true = np.array(all_labels[asp])
        y_pred = np.array(all_preds[asp])
        none_i = NONE_IDX[asp]

        # Detection F1 (semua kelas)
        detect_f1[asp] = f1_score(y_true, y_pred, average='macro',
                                   zero_division=0)

        # Sentiment F1 (hanya sampel bukan None)
        mask = y_true != none_i
        if mask.sum() > 0:
            sentiment_f1[asp] = f1_score(
                y_true[mask], y_pred[mask],
                average='macro', zero_division=0
            )
        else:
            sentiment_f1[asp] = 0.0

    # Rata-rata tertimbang berdasarkan jumlah sampel aktif (non-None)
    weights_sent = [
        sum(1 for l in all_labels[a] if l != NONE_IDX[a])
        for a in ASPECTS
    ]
    weights_det  = [len(all_labels[a]) for a in ASPECTS]

    avg_detect   = float(np.average(list(detect_f1.values()),
                                    weights=weights_det)) \
                   if sum(weights_det) > 0 else 0.0
    avg_sentiment= float(np.average(list(sentiment_f1.values()),
                                    weights=weights_sent)) \
                   if sum(weights_sent) > 0 else 0.0

    return detect_f1, sentiment_f1, avg_detect, avg_sentiment, \
           all_preds, all_labels



# ── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(config=CONFIG):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    os.makedirs(config['save_dir'], exist_ok=True)

    df_train = pd.read_csv('sample_data/df_train.csv')
    df_val   = pd.read_csv('sample_data/df_val.csv')


    tokenizer    = AutoTokenizer.from_pretrained(config['model_name'])
    train_ds     = ABSADataset(df_train, tokenizer, config['max_length'], ASPECTS)
    val_ds       = ABSADataset(df_val,   tokenizer, config['max_length'], ASPECTS)
    collator     = ABSACollator(tokenizer)
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                              shuffle=True,  num_workers=2, pin_memory=True,
                              collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=config['batch_size'],
                              shuffle=False, num_workers=2, pin_memory=True,
                              collate_fn=collator)

    model = ABSAModel(config['model_name'], ASPECTS,
                      NUM_CLASSES, config['dropout_rate']).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")

    optimizer    = AdamW(model.parameters(), lr=config['learning_rate'],
                         weight_decay=config['weight_decay'])
    total_steps  = len(train_loader) * config['num_epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    best_f1, patience_cnt = 0.0, 0
    history = {'train_loss': [], 'detect_f1': [], 'sentiment_f1': [],
               'detect_per_asp': [], 'sentiment_per_asp': []}

    print(f"\n{'='*60}")
    print(f"TRAINING — Skema 4-kelas (Neg/Neu/Pos/None)")
    print(f"Train: {len(df_train)} | Val: {len(df_val)}")
    print(f"{'='*60}\n")

    for epoch in range(config['num_epochs']):
        model.train()
        ep_loss, n_batch = 0.0, 0

        for step, batch in enumerate(train_loader):
            iids  = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            labs  = {a: batch['labels'][a] for a in ASPECTS}

            optimizer.zero_grad()
            logits = model(iids, amask)
            loss   = compute_loss(logits, labs, CLASS_WEIGHTS, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n_batch += 1
            if (step+1) % 50 == 0:
                print(f"  Ep{epoch+1} step{step+1}/{len(train_loader)} "
                      f"loss={ep_loss/n_batch:.4f}")

        avg_loss = ep_loss / n_batch
        det_f1, sent_f1, avg_det, avg_sent, _, _ = evaluate(
            model, val_loader, device)

        history['train_loss'].append(avg_loss)
        history['detect_f1'].append(avg_det)
        history['sentiment_f1'].append(avg_sent)
        history['detect_per_asp'].append(det_f1)
        history['sentiment_per_asp'].append(sent_f1)

        print(f"\nEpoch {epoch+1}/{config['num_epochs']} | Loss: {avg_loss:.4f}")
        print(f"  Detection F1   (incl-None) : {avg_det:.4f}")
        print(f"  Sentiment F1   (excl-None) : {avg_sent:.4f}  ← metrik utama")
        print(f"  {'Aspek':<30} {'Detect':>8} {'Sentimen':>10}")
        print(f"  {'─'*50}")
        for asp in ASPECTS:
            print(f"  {asp:<30} {det_f1[asp]:>8.4f} {sent_f1[asp]:>10.4f}")
        print()

        # Early stopping berbasis Sentiment F1
        if avg_sent > best_f1:
            best_f1, patience_cnt = avg_sent, 0
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(),
                        'val_sent_f1': best_f1, 'sent_f1_per_asp': sent_f1,
                        'det_f1_per_asp': det_f1},
                       os.path.join(config['save_dir'], 'best_model.pt'))
            print(f"  ✅ Best model saved (Sentiment F1: {best_f1:.4f})\n")
        else:
            patience_cnt += 1
            print(f"  ⏳ No improvement. Patience: {patience_cnt}/{config['patience']}\n")
            if patience_cnt >= config['patience']:
                print(f"  🛑 Early stopping at epoch {epoch+1}")
                break

    pickle.dump(history,
                open(os.path.join(config['save_dir'],'history.pkl'),'wb'))
    print(f"\n{'='*60}")
    print(f"SELESAI | Best Sentiment F1: {best_f1:.4f}")
    print(f"{'='*60}\n")
    return model, history


# ── EVALUASI TEST ──────────────────────────────────────────────────────────────
def evaluate_test(config=CONFIG):
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_test   = pd.read_csv('sample_data/df_test.csv')
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    test_ds   = ABSADataset(df_test, tokenizer, config['max_length'], ASPECTS)
    collator  = ABSACollator(tokenizer)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                             shuffle=False, num_workers=2, collate_fn=collator)

    model = ABSAModel(config['model_name'], ASPECTS,
                      NUM_CLASSES, config['dropout_rate']).to(device)
    ckpt  = torch.load(os.path.join(config['save_dir'],'best_model.pt'),
                       map_location=device)
    model.load_state_dict(ckpt['model_state'])

    det_f1, sent_f1, avg_det, avg_sent, all_preds, all_labels = evaluate(
        model, test_loader, device)

    print(f"\n{'='*60}")
    print("HASIL TEST SET")
    print(f"{'='*60}")
    print(f"\n  {'Aspek':<30} {'Detection':>10} {'Sentimen':>10}")
    print(f"  {'─'*52}")
    for asp in ASPECTS:
        print(f"  {asp:<30} {det_f1[asp]:>10.4f} {sent_f1[asp]:>10.4f}")
    print(f"  {'─'*52}")
    print(f"  {'Rata-rata (tertimbang)':<30} {avg_det:>10.4f} {avg_sent:>10.4f}")

    print(f"\n\n── Classification Report per Aspek ──\n")
    for asp in ASPECTS:
        y_true  = np.array(all_labels[asp])
        y_pred  = np.array(all_preds[asp])
        none_i  = NONE_IDX[asp]
        n_aktif = (y_true != none_i).sum()

        print(f"{'─'*40}")
        print(f"Aspek: {asp}  (n_total={len(y_true)}, n_aktif={n_aktif})")
        print(classification_report(
            y_true, y_pred, target_names=LABEL_NAMES[asp], zero_division=0))

    return det_f1, sent_f1


# ── INFERENCE ──────────────────────────────────────────────────────────────────
def predict(text, model, tokenizer, device, config=CONFIG):
    import re
    text = text.lower()
    text = re.sub(r'http\S+|www\S+','',text)
    text = re.sub(r'\s+',' ',text).strip()

    text = preprocess_text(text)
    enc   = tokenizer(text, max_length=config['max_length'],
                      padding='max_length', truncation=True,
                      return_tensors='pt')
    iids  = enc['input_ids'].to(device)
    amask = enc['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        logits = model(iids, amask)

    results = {}
    for asp in ASPECTS:
        probs   = torch.softmax(logits[asp], dim=-1).squeeze(0).cpu().numpy()
        pred_c  = int(probs.argmax())
        results[asp] = {
            'label'     : LABEL_NAMES[asp][pred_c],
            'confidence': float(probs[pred_c]),
            'probs'     : {LABEL_NAMES[asp][i]: round(float(probs[i]),4)
                           for i in range(len(LABEL_NAMES[asp]))},
        }
    return results


# ── MAIN ────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, history = train()
    evaluate_test()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
    print("\n=== CONTOH INFERENCE ===\n")
    tests = [
        "mantap",
        "beritanya akurat dan terpercaya, mantap",
        "terlalu mahal, 50rb per bulan itu kemahalan banget",
        "aplikasi sering force close, tidak bisa dibuka",
        "sudah berlangganan tapi tidak bisa login sama sekali",
        # Multi-aspek
        "beritanya bagus, tapi harga langgannya mahal",
        "tampilan bagus tapi sering crash",
        "berita terpercaya, harga worth it, tapi fitur kliping tidak berfungsi",
    ]
    for text in tests:
        print(f"Input: '{text}'")
        res = predict(text, model, tokenizer, device)
        for asp, r in res.items():
            label = r['label']
            conf  = r['confidence']
            if label != 'None':
                print(f"  {asp:<30}: {label} ({conf:.3f})")
            else:
                print(f"  {asp:<30}: None ({conf:.3f})")
        print()