import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from preprocessing_functions import FINAL_ASPECTS, NUM_CLASSES, LABEL_NAMES

NONE_IDX = {asp: NUM_CLASSES[asp] - 1 for asp in FINAL_ASPECTS}


def _asp_key(asp: str) -> str:
    """Konversi nama aspek ke format aman untuk MLflow metric key."""
    return (
        asp.replace(' ', '_')
           .replace('&', 'and')
           .replace('/', '_')
           .lower()
    )


def _eval_loop(model, loader, device):
    """
    Loop evaluasi inti — dapat digunakan saat validasi per-epoch maupun evaluasi
    final pada test set.

    Returns
    -------
    detect_f1    : dict  — macro F1 semua kelas (termasuk None) per aspek
    sentiment_f1 : dict  — macro F1 kelas sentimen saja (tidak termasuk None) per aspek
    avg_detect   : float — rata-rata tertimbang detect_f1
    avg_sentiment: float — rata-rata tertimbang sentiment_f1
    all_preds    : dict  — prediksi per aspek
    all_labels   : dict  — label asli per aspek
    """
    model.eval()
    all_preds  = {a: [] for a in FINAL_ASPECTS}
    all_labels = {a: [] for a in FINAL_ASPECTS}

    with torch.no_grad():
        for batch in loader:
            iids  = batch['input_ids'].to(device)
            amask = batch['attention_mask'].to(device)
            logits = model(iids, amask)
            for asp in FINAL_ASPECTS:
                preds  = logits[asp].argmax(dim=-1).cpu().numpy()
                labels = batch['labels'][asp].numpy()
                all_preds[asp].extend(preds.tolist())
                all_labels[asp].extend(labels.tolist())

    detect_f1    = {}
    sentiment_f1 = {}

    for asp in FINAL_ASPECTS:
        y_true = np.array(all_labels[asp])
        y_pred = np.array(all_preds[asp])
        none_i = NONE_IDX[asp]

        # Detection F1: semua kelas termasuk None
        detect_f1[asp] = f1_score(y_true, y_pred, average='macro', zero_division=0)

        # Sentiment F1: hanya sampel dengan label bukan None
        mask = y_true != none_i
        sentiment_f1[asp] = (
            f1_score(y_true[mask], y_pred[mask], average='macro', zero_division=0)
            if mask.sum() > 0 else 0.0
        )

    n_det  = [len(all_labels[a]) for a in FINAL_ASPECTS]
    n_sent = [sum(1 for l in all_labels[a] if l != NONE_IDX[a]) for a in FINAL_ASPECTS]

    avg_detect    = float(np.average(list(detect_f1.values()), weights=n_det))
    avg_sentiment = (
        float(np.average(list(sentiment_f1.values()), weights=n_sent))
        if sum(n_sent) > 0 else 0.0
    )

    return detect_f1, sentiment_f1, avg_detect, avg_sentiment, all_preds, all_labels


def evaluate_model(config: dict, trained: dict, data: dict) -> dict:
    """
    Evaluasi model pada test set dan kembalikan metrik dalam format flat
    yang siap di-log ke MLflow.

    Metrik yang dikembalikan:
      test_mean_detect_f1        — rata-rata Detection F1 semua aspek
      test_mean_sentiment_f1     — rata-rata Sentiment F1 semua aspek (metrik utama)
      test_{asp}_detect_f1       — Detection F1 per aspek
      test_{asp}_sentiment_f1    — Sentiment F1 per aspek
    """
    model_type = config['model']['type']
    if model_type == 'indobert_multitask':
        return _evaluate_indobert(config, trained, data)
    raise ValueError(f"Model type tidak didukung: {model_type}")


def _evaluate_indobert(config: dict, trained: dict, data: dict) -> dict:
    from torch.utils.data import DataLoader
    from train import ABSADataset, ABSACollator

    rep_cfg = config['representation']
    params  = config['model']['params']
    model     = trained['model']      # best model state (sudah dimuat di train_model)
    tokenizer = trained['tokenizer']
    device    = trained['device']

    test_ds     = ABSADataset(data['df_test'], tokenizer, rep_cfg['max_length'], FINAL_ASPECTS)
    collator    = ABSACollator(tokenizer)
    test_loader = DataLoader(
        test_ds, batch_size=params['batch_size'],
        shuffle=False, num_workers=0, collate_fn=collator,
    )

    det_f1, sent_f1, avg_det, avg_sent, all_preds, all_labels = _eval_loop(
        model, test_loader, device
    )

    # ── Cetak laporan ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("HASIL EVALUASI TEST SET")
    print(f"{'='*60}")
    print(f"{'Aspek':<30} {'Detection':>10} {'Sentimen':>10}")
    print(f"{'─'*52}")
    for asp in FINAL_ASPECTS:
        print(f"{asp:<30} {det_f1[asp]:>10.4f} {sent_f1[asp]:>10.4f}")
    print(f"{'─'*52}")
    print(f"{'Rata-rata (tertimbang)':<30} {avg_det:>10.4f} {avg_sent:>10.4f}")
    print(f"\n  Sentiment F1 adalah metrik utama keberhasilan model.")

    print(f"\n── Classification Report per Aspek ──")
    for asp in FINAL_ASPECTS:
        y_true  = np.array(all_labels[asp])
        y_pred  = np.array(all_preds[asp])
        n_aktif = (y_true != NONE_IDX[asp]).sum()
        print(f"\n{asp}  (n_total={len(y_true)}, n_aktif={n_aktif})")
        print(classification_report(
            y_true, y_pred, target_names=LABEL_NAMES[asp], zero_division=0,
        ))

    # ── Metrik flat untuk MLflow ────────────────────────────────────────
    metrics = {
        'test_mean_detect_f1'   : avg_det,
        'test_mean_sentiment_f1': avg_sent,
    }
    for asp in FINAL_ASPECTS:
        k = _asp_key(asp)
        metrics[f'test_{k}_detect_f1']    = det_f1[asp]
        metrics[f'test_{k}_sentiment_f1'] = sent_f1[asp]

    return metrics
