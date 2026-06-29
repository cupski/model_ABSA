import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import mlflow

from model.absa_model import ABSADataset, ABSACollator, ABSAModel, compute_loss
from pipeline.evaluate_model import _eval_loop
from preprocessing.preprocessing_functions import FINAL_ASPECTS, NUM_CLASSES


def _compute_val_loss(model, loader, class_weights, device) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            iids   = batch['input_ids'].to(device)
            amask  = batch['attention_mask'].to(device)
            labs   = {a: batch['labels'][a] for a in FINAL_ASPECTS}
            logits = model(iids, amask)
            total += compute_loss(logits, labs, class_weights, device).item()
            n += 1
    return total / n if n > 0 else 0.0


def train_model(config: dict, data: dict) -> dict:
    """
    Muat dan latih model sesuai konfigurasi.

    Parameters
    ----------
    config : dict  — konfigurasi eksperimen dari YAML
    data   : dict  — output dari prepare_data() berisi df_train, df_val, class_weights

    Returns
    -------
    dict dengan kunci:
      model           : model terbaik (best checkpoint sudah dimuat kembali)
      tokenizer       : tokenizer yang digunakan
      history         : riwayat metrik per epoch
      best_val_f1     : Sentiment F1 terbaik pada validation set
      best_val_det_f1 : Detection F1 pada epoch terbaik
      save_dir        : direktori penyimpanan checkpoint
      device          : torch.device yang digunakan
    """
    model_type = config['model']['type']
    if model_type == 'indobert_multitask':
        return _train_indobert(config, data)
    raise ValueError(f"Model type tidak didukung: {model_type}")


def _train_indobert(config: dict, data: dict) -> dict:
    rep_cfg   = config['representation']
    model_cfg = config['model']
    params    = model_cfg['params']
    save_dir  = model_cfg['save_dir']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    os.makedirs(save_dir, exist_ok=True)

    df_train = data['df_train']
    df_val   = data['df_val']

    class_weights = (
        data['class_weights']
        if model_cfg.get('use_class_weights', True)
        else {asp: [1.0] * NUM_CLASSES[asp] for asp in FINAL_ASPECTS}
    )

    tokenizer = AutoTokenizer.from_pretrained(
        rep_cfg['model_name'],
        revision=rep_cfg.get('model_revision', 'main'),
    )
    collator  = ABSACollator(tokenizer)
    train_ds  = ABSADataset(df_train, tokenizer, rep_cfg['max_length'], FINAL_ASPECTS)
    val_ds    = ABSADataset(df_val,   tokenizer, rep_cfg['max_length'], FINAL_ASPECTS)

    # num_workers=0 diperlukan di Windows untuk menghindari masalah multiprocessing
    train_loader      = DataLoader(train_ds, batch_size=params['batch_size'],
                                   shuffle=True,  num_workers=0, collate_fn=collator)
    train_eval_loader = DataLoader(train_ds, batch_size=params['batch_size'],
                                   shuffle=False, num_workers=0, collate_fn=collator)
    val_loader        = DataLoader(val_ds,   batch_size=params['batch_size'],
                                   shuffle=False, num_workers=0, collate_fn=collator)

    model = ABSAModel(
        rep_cfg['model_name'], FINAL_ASPECTS,
        NUM_CLASSES, params['dropout_rate'],
    ).to(device)
    print(f"  Parameter: {sum(p.numel() for p in model.parameters()):,}")

    optimizer    = AdamW(model.parameters(),
                         lr=params['learning_rate'],
                         weight_decay=params['weight_decay'])
    total_steps  = len(train_loader) * params['num_epochs']
    warmup_steps = int(total_steps * params['warmup_ratio'])
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_sent, best_det, patience_cnt = 0.0, 0.0, 0
    history = {
        'train_loss'              : [],
        'val_loss'                : [],
        'train_avg_detection_f1'  : [],
        'val_avg_detection_f1'    : [],
        'train_avg_sentiment_f1'  : [],
        'val_avg_sentiment_f1'    : [],
        'val_sentiment_f1_per_asp': [],
        'val_detect_f1_per_asp'   : [],
    }

    print(f"\n  {'='*56}")
    print(f"  TRAINING IndoBERT — {len(df_train)} train | {len(df_val)} val")
    print(f"  Epochs: {params['num_epochs']} | Batch: {params['batch_size']} | "
          f"LR: {params['learning_rate']:.2e}")
    print(f"  {'='*56}\n")

    for epoch in range(params['num_epochs']):
        model.train()
        ep_loss, n_batch = 0.0, 0

        for step, batch in enumerate(train_loader):
            iids  = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            labs  = {a: batch['labels'][a] for a in FINAL_ASPECTS}

            optimizer.zero_grad()
            logits = model(iids, amask)
            loss   = compute_loss(logits, labs, class_weights, device)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), params['max_grad_norm'])
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            n_batch += 1

            if (step + 1) % 50 == 0:
                print(f"    Ep{epoch+1} step{step+1}/{len(train_loader)} "
                      f"loss={ep_loss/n_batch:.4f}")

        # ── Hitung semua metrik setelah epoch selesai ──────────────────
        avg_train_loss = ep_loss / n_batch
        avg_val_loss   = _compute_val_loss(model, val_loader, class_weights, device)

        _, _, avg_tr_det, avg_tr_sent, _, _ = _eval_loop(model, train_eval_loader, device)
        det_f1, sent_f1, avg_det, avg_sent, _, _ = _eval_loop(model, val_loader, device)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_avg_detection_f1'].append(avg_tr_det)
        history['val_avg_detection_f1'].append(avg_det)
        history['train_avg_sentiment_f1'].append(avg_tr_sent)
        history['val_avg_sentiment_f1'].append(avg_sent)
        history['val_sentiment_f1_per_asp'].append(dict(sent_f1))
        history['val_detect_f1_per_asp'].append(dict(det_f1))

        # ── Log metrik per epoch ke MLflow ────────────────────────────
        mlflow.log_metric('train_loss',             avg_train_loss, step=epoch + 1)
        mlflow.log_metric('val_loss',               avg_val_loss,   step=epoch + 1)
        mlflow.log_metric('train_avg_detection_f1', avg_tr_det,     step=epoch + 1)
        mlflow.log_metric('val_avg_detection_f1',   avg_det,        step=epoch + 1)
        mlflow.log_metric('train_avg_sentiment_f1', avg_tr_sent,    step=epoch + 1)
        mlflow.log_metric('val_avg_sentiment_f1',   avg_sent,       step=epoch + 1)

        print(f"  Epoch {epoch+1}/{params['num_epochs']} | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Det: {avg_det:.4f} | Sent: {avg_sent:.4f}  <- metrik utama")

        if avg_sent > best_sent:
            best_sent, best_det, patience_cnt = avg_sent, avg_det, 0
            torch.save({
                'epoch'          : epoch + 1,
                'model_state'    : model.state_dict(),
                'val_sent_f1'    : best_sent,
                'val_det_f1'     : best_det,
                'sent_f1_per_asp': sent_f1,
                'det_f1_per_asp' : det_f1,
                'config'         : config,
            }, os.path.join(save_dir, 'best_model.pt'))
            print(f"    Best model tersimpan (Val Sentiment F1: {best_sent:.4f})")
        else:
            patience_cnt += 1
            if patience_cnt >= params['patience']:
                print(f"    Early stopping di epoch {epoch+1}")
                break

    pickle.dump(history, open(os.path.join(save_dir, 'history.pkl'), 'wb'))

    ckpt = torch.load(os.path.join(save_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"\n  Best model dimuat (dari epoch {ckpt['epoch']}, "
          f"Val Sentiment F1: {best_sent:.4f})")

    return {
        'model'          : model,
        'tokenizer'      : tokenizer,
        'history'        : history,
        'best_val_f1'    : best_sent,
        'best_val_det_f1': best_det,
        'save_dir'       : save_dir,
        'device'         : device,
    }


# ── PENGUJIAN MODUL ───────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

    # Smoke test: jalankan 1 epoch pada subset kecil untuk memverifikasi alur kerja
    SMOKE_CONFIG = {
        'data': {
            'path'       : 'ABSA_dataset_final_CLEAN.csv',
            'text_column': 'Komentar',
            'split': {'train_ratio': 0.70, 'val_ratio': 0.15, 'random_state': 42},
        },
        'preprocessing': {
            'remove_emoji': True, 'lowercase': True, 'remove_url_mention': True,
            'compress_repeated_chars': True, 'remove_special_chars': True,
            'normalize_slang': True, 'remove_stopwords': False,
        },
        'representation': {
            'type': 'indobert',
            'model_name'    : 'indobenchmark/indobert-base-p1',
            'model_revision': 'main',
            'max_length'    : 128,
        },
        'model': {
            'type'             : 'indobert_multitask',
            'use_class_weights': True,
            'save_dir'         : 'model_output_smoke',
            'params': {
                'dropout_rate' : 0.1,
                'batch_size'   : 4,
                'learning_rate': 2e-5,
                'num_epochs'   : 1,
                'warmup_ratio' : 0.1,
                'weight_decay' : 0.01,
                'max_grad_norm': 1.0,
                'patience'     : 1,
            },
        },
    }

    print("=" * 60)
    print("SMOKE TEST train_model (1 epoch, subset kecil)")
    print("=" * 60)

    from pipeline.prepare_data import prepare_data
    data = prepare_data(SMOKE_CONFIG)

    data['df_train'] = data['df_train'].head(20)
    data['df_val']   = data['df_val'].head(10)

    trained = train_model(SMOKE_CONFIG, data)
    print(f"\nSmoke test selesai. Best Val Sentiment F1: {trained['best_val_f1']:.4f}")
    print("=" * 60)
