"""
Tahap Pelatihan Model (Workflow Wrapper)
=========================================
Membungkus pipeline/train_model.py menjadi satu unit eksekusi yang dapat
diisolasi dan diulang secara independen dalam automated ML workflow pipeline.

Tanggung jawab tambahan dibanding pipeline/train_model.py:
  - Membuka sesi MLflow run untuk mencatat seluruh metadata eksperimen
  - Mengembalikan dict ringan (tanpa objek model) agar dapat disimpan sebagai
    Metaflow artifact lintas step — model disimpan ke disk oleh pipeline/train_model.py
    dan dimuat ulang oleh tahap evaluate_model.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
import mlflow.pytorch

from model.absa_model import set_seed
from pipeline.train_model import train_model
from run_experiment import flatten_config, get_git_commit


def run_train_model(model_config: dict, data: dict) -> dict:
    """
    Latih model dan catat eksperimen ke MLflow.

    Membuka satu MLflow run yang akan dilanjutkan oleh tahap evaluate_model
    (via mlflow.start_run(run_id=...)) untuk mencatat metrik test set dalam
    run yang sama.

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen
    data         : dict — output run_prepare_data() (df_train, df_val, df_test, class_weights)

    Returns
    -------
    dict:
      run_id          : str   — MLflow run ID (untuk dilanjutkan di evaluate_model)
      save_dir        : str   — direktori checkpoint model
      best_val_f1     : float — Sentiment F1 terbaik pada validation set
      best_val_det_f1 : float — Detection F1 pada epoch terbaik
    """
    mlflow_cfg   = model_config.get('mlflow', {})
    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        mlflow_cfg.get('tracking_uri', 'http://localhost:5000'),
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(model_config['experiment']['name'])

    os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = 'true'

    seed = model_config['experiment'].get('seed', 42)
    set_seed(seed)

    run_name = model_config['experiment'].get('run_name', model_config['experiment']['name'])

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Catat metadata versi
        mlflow.set_tag('git_commit',     get_git_commit())
        mlflow.set_tag('model_name',     model_config['representation']['model_name'])
        mlflow.set_tag('model_revision', model_config['representation'].get('model_revision', 'main'))
        mlflow.set_tag('mlflow.note.content', model_config['experiment'].get('description', ''))
        mlflow.set_tag('triggered_by', os.environ.get('ABSA_TRIGGER_REASON', 'scheduled'))

        mlflow.log_param('experiment.seed', seed)
        for k, v in flatten_config(model_config).items():
            mlflow.log_param(k, str(v)[:500])

        # Latih model — pipeline/train_model.py mencatat metrik per epoch ke run aktif
        print(f"\n  MLflow Run ID: {run_id}")
        trained = train_model(model_config, data)

        mlflow.log_metric('best_val_sentiment_f1', trained['best_val_f1'])
        mlflow.log_metric('best_val_detection_f1', trained['best_val_det_f1'])

        # Log artefak kecil (bukan checkpoint .pt)
        _LARGE_EXTS = {'.bin', '.safetensors', '.pt', '.pth'}
        save_dir = trained['save_dir']
        if os.path.isdir(save_dir):
            for fname in os.listdir(save_dir):
                fpath = os.path.join(save_dir, fname)
                if os.path.isfile(fpath) and os.path.splitext(fname)[1].lower() not in _LARGE_EXTS:
                    mlflow.log_artifact(fpath, artifact_path='model_artifacts')

    print(f"  Pelatihan selesai. Best Val Sentiment F1: {trained['best_val_f1']:.4f}")

    # Kembalikan hanya metadata yang dapat diserialisasi Metaflow (bukan objek model)
    return {
        'run_id'         : run_id,
        'save_dir'       : save_dir,
        'best_val_f1'    : trained['best_val_f1'],
        'best_val_det_f1': trained['best_val_det_f1'],
    }
