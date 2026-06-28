import os
import json
import argparse
import subprocess

import yaml
import mlflow
import mlflow.pytorch

from pipeline.validate_data  import validate_data
from pipeline.prepare_data   import prepare_data
from pipeline.train_model    import train_model
from pipeline.evaluate_model import evaluate_model


# ── UTILITAS ──────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def get_git_commit() -> str:
    """Ambil commit hash HEAD sebagai pointer versi kode."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], text=True
        ).strip()
    except Exception:
        return 'unknown'


def flatten_config(cfg: dict, prefix: str = '') -> dict:
    """
    Ratakan dict konfigurasi bersarang menjadi dict satu level
    agar dapat di-log ke MLflow sebagai params (nilai harus skalar).
    """
    out = {}
    for k, v in cfg.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_config(v, key))
        elif isinstance(v, list):
            out[key] = str(v)
        else:
            out[key] = v
    return out


# ── ORKESTRASI PIPELINE ────────────────────────────────────────────────────────

def run_experiment(config_path: str) -> dict:
    """
    Jalankan satu eksperimen lengkap dari berkas konfigurasi:
      1. Muat konfigurasi
      2. Buka sesi MLflow dan catat seluruh metadata
      3. Validasi data
      4. Persiapan data
      5. Pelatihan model
      6. Evaluasi model
      7. Catat metrik dan simpan artefak ke MLflow
      8. Tutup sesi MLflow

    Parameters
    ----------
    config_path : str — path ke berkas YAML konfigurasi eksperimen

    Returns
    -------
    dict — metrik evaluasi test set
    """
    config = load_config(config_path)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", config["mlflow"]["tracking_uri"])
    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(config['experiment']['name'])

    with mlflow.start_run(run_name=config['experiment']['name']) as run:
        run_id = run.info.run_id

        print(f"\n{'='*60}")
        print(f"EKSPERIMEN: {config['experiment']['name']}")
        print(f"Deskripsi : {config['experiment'].get('description', '-')}")
        print(f"MLflow Run ID: {run_id}")
        print(f"{'='*60}")

        # ── Catat metadata versi ───────────────────────────────────
        git_commit = get_git_commit()
        mlflow.set_tag('git_commit - versi kode dan data',       git_commit)
        mlflow.log_artifact("data/raw/ABSA_dataset_final_CLEAN.csv.dvc", artifact_path="dataset")
        mlflow.set_tag('model_name',       config['representation']['model_name'])
        mlflow.set_tag('model_revision',   config['representation'].get('model_revision', 'main'))
        mlflow.set_tag('description',      config['experiment'].get('description', ''))
        mlflow.log_artifact("requirements.txt")
        print(f"\nVersi kode dan data : {git_commit[:12]}")
        print(f"Model       : {config['representation']['model_name']} "
              f"@ {config['representation'].get('model_revision', 'main')}")

        # ── Catat seluruh parameter konfigurasi ───────────────────
        for k, v in flatten_config(config).items():
            mlflow.log_param(k, str(v)[:500])

        # ── 1. Validasi data ───────────────────────────────────────
        print("\n[1/4] Validasi data...")
        val_report = validate_data(config)
        if not val_report['passed']:
            raise RuntimeError(f"Validasi data gagal: {val_report['issues']}")
        print(f"  OK — {val_report['total_rows']} baris, semua pemeriksaan lulus")
        if val_report['issues']:
            print(f"  Peringatan: {val_report['issues']}")
        mlflow.log_param('data.n_rows', val_report['total_rows'])

        # ── 2. Persiapan data ──────────────────────────────────────
        print("\n[2/4] Persiapan data...")
        data = prepare_data(config)
        n_train = len(data['df_train'])
        n_val   = len(data['df_val'])
        n_test  = len(data['df_test'])
        print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
        mlflow.log_param('data.n_train', n_train)
        mlflow.log_param('data.n_val',   n_val)
        mlflow.log_param('data.n_test',  n_test)

        # Simpan class weights sebagai artefak
        save_dir = config['model']['save_dir']
        os.makedirs(save_dir, exist_ok=True)
        cw_path = os.path.join(save_dir, 'class_weights.json')
        with open(cw_path, 'w', encoding='utf-8') as f:
            json.dump(data['class_weights'], f, indent=2, ensure_ascii=False)

        # ── 3. Pelatihan model ─────────────────────────────────────
        print("\n[3/4] Pelatihan model...")
        trained = train_model(config, data)
        mlflow.log_metric('val_best_sentiment_f1', trained['best_val_f1'])

        # ── 4. Evaluasi model ──────────────────────────────────────
        print("\n[4/4] Evaluasi model pada test set...")
        metrics = evaluate_model(config, trained, data)
        mlflow.log_metrics(metrics)

        # ── Simpan artefak ke MLflow ────────────────────────────────
        # Checkpoint model, history, class weights
        if os.path.isdir(save_dir):
            mlflow.log_artifacts(save_dir, artifact_path='model_artifacts')

        # File konfigurasi yang digunakan eksperimen ini
        mlflow.log_artifact(config_path, artifact_path='config')

        print(f"\n{'='*60}")
        print("EKSPERIMEN SELESAI")
        print(f"{'='*60}")
        print(f"  Test Mean Sentiment F1 : {metrics.get('test_mean_sentiment_f1', 0):.4f}  <- metrik utama")
        print(f"  Test Mean Detection F1 : {metrics.get('test_mean_detect_f1', 0):.4f}")
        print(f"  MLflow Run ID          : {run_id}")
        print(f"  Jalankan: mlflow ui    untuk melihat hasil eksperimen")
        print(f"{'='*60}\n")

    return metrics


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Jalankan satu eksperimen ABSA dari berkas konfigurasi YAML.')
    parser.add_argument(
        'config',
        type=str,
        help='Path ke berkas konfigurasi YAML (misal: configs/experiment_indobert_baseline.yaml)',
    )
    args = parser.parse_args()

    run_experiment(args.config)
