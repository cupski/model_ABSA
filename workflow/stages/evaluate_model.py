"""
Tahap Evaluasi Model (Workflow Wrapper)
=========================================
Membungkus pipeline/evaluate_model.py menjadi satu unit eksekusi yang dapat
diisolasi dan diulang secara independen dalam automated ML workflow pipeline.

Tanggung jawab tambahan dibanding pipeline/evaluate_model.py:
  - Memuat ulang model dari checkpoint disk (save_dir/best_model.pt) karena
    objek model tidak dapat dilewatkan antar Metaflow step sebagai artifact
  - Melanjutkan MLflow run yang dibuka oleh tahap train_model untuk mencatat
    metrik test set dalam run yang sama
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import mlflow

from model.checkpoint_io import load_model_from_checkpoint
from pipeline.evaluate_model import evaluate_model


def run_evaluate_model(model_config: dict, train_result: dict, data: dict) -> dict:
    """
    Muat ulang model dari checkpoint, evaluasi pada test set, dan catat
    metrik ke MLflow run yang sama dengan tahap train_model.

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen
    train_result : dict — output run_train_model() (run_id, save_dir, ...)
    data         : dict — output run_prepare_data() (df_train, df_val, df_test, ...)

    Returns
    -------
    dict — metrik evaluasi test set (siap di-log ke MLflow)
    """
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = train_result['save_dir']
    run_id   = train_result['run_id']

    print(f"  Memuat model dari: {save_dir}/best_model.pt")
    model, tokenizer, _ = load_model_from_checkpoint(save_dir, device)

    trained_reconstructed = {
        'model'    : model,
        'tokenizer': tokenizer,
        'device'   : device,
        'save_dir' : save_dir,
    }

    metrics = evaluate_model(model_config, trained_reconstructed, data)

    # Lanjutkan MLflow run yang sama untuk mencatat metrik test set
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        mlflow_cfg.get('tracking_uri', 'http://localhost:5000'),
    )
    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
        # Log artefak evaluasi yang dihasilkan pipeline/evaluate_model.py
        for fname in ('classification_report.txt', 'confusion_matrix.png'):
            fpath = os.path.join(save_dir, fname)
            if os.path.isfile(fpath):
                mlflow.log_artifact(fpath, artifact_path='model_artifacts')

    print(f"  Test Mean Sentiment F1: {metrics.get('test_mean_sentiment_f1', 0):.4f} ← metrik utama")
    return metrics
