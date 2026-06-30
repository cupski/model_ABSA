"""
Tahap Validasi Model
=====================
Komponen baru dalam automated ML workflow pipeline.

Memastikan model yang baru dilatih memenuhi dua pengujian sebelum
dapat diteruskan ke tahap registrasi:

  Uji 1 — Threshold absolut:
    Metrik test set harus memenuhi nilai minimum yang ditetapkan di
    workflow/pipeline_config.yaml (model_validation.min_sentiment_f1,
    model_validation.min_detection_f1).

  Uji 2 — Perbandingan dengan model produksi:
    Jika require_improvement=True, model baru harus lebih baik dari
    model yang sedang berjalan di produksi (berdasarkan comparison_metric).

    Mode Bootstrap (belum ada model produksi):
    Saat registry MLflow belum memiliki registered model ini sama sekali
    (belum pernah ada eksperimen yang didaftarkan) ATAU belum ada versi
    berstage production_stage, uji 2 berada dalam kondisi "belum dapat
    dibandingkan". Perilaku dalam kondisi ini dikendalikan oleh
    model_validation.skip_comparison_if_no_production di
    workflow/pipeline_config.yaml:
      True  → uji 2 dilewati otomatis dan dianggap lulus (bootstrap)
      False → uji 2 dianggap GAGAL — memaksa registrasi model produksi
              pertama secara manual sebelum pipeline dapat lolos penuh
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import mlflow
from mlflow.exceptions import MlflowException
from mlflow import MlflowClient


# ── Uji 1: Threshold Absolut ──────────────────────────────────────────────────

def _check_threshold(metrics: dict, validation_cfg: dict) -> dict:
    """Periksa apakah metrik test set memenuhi nilai minimum yang ditetapkan."""
    min_sent = validation_cfg['min_sentiment_f1']
    min_det  = validation_cfg['min_detection_f1']

    actual_sent = metrics.get('test_mean_sentiment_f1', 0.0)
    actual_det  = metrics.get('test_mean_detect_f1', 0.0)

    passed = (actual_sent >= min_sent) and (actual_det >= min_det)

    reasons = []
    if actual_sent < min_sent:
        reasons.append(
            f"Sentiment F1 ({actual_sent:.4f}) < threshold ({min_sent})"
        )
    if actual_det < min_det:
        reasons.append(
            f"Detection F1 ({actual_det:.4f}) < threshold ({min_det})"
        )

    return {
        'passed' : passed,
        'reasons': reasons,
        'details': {
            'actual_sentiment_f1'  : actual_sent,
            'actual_detection_f1'  : actual_det,
            'min_sentiment_f1'     : min_sent,
            'min_detection_f1'     : min_det,
        },
    }


# ── Uji 2: Perbandingan dengan Model Produksi ─────────────────────────────────

def _get_production_metric(workflow_config: dict, model_config: dict) -> dict:
    """
    Ambil nilai metrik perbandingan dari model yang sedang berjalan di produksi
    melalui MLflow Model Registry.

    Membedakan dua kondisi "tidak ada nilai" agar fallback bootstrap dapat
    diputuskan secara eksplisit, bukan disamaratakan sebagai error generik:
      - registered model belum pernah dibuat sama sekali di registry
        (RESOURCE_DOES_NOT_EXIST — kondisi wajar untuk eksperimen pertama)
      - registered model sudah ada, tapi belum ada versi berstage produksi

    Returns
    -------
    dict:
      exists : bool          — True jika ditemukan versi model di stage produksi
      value  : float | None  — nilai metrik perbandingan (None jika exists=False)
      reason : str           — penjelasan kondisi untuk logging
    """
    mlflow_wf  = workflow_config.get('mlflow', {})
    mlflow_mdl = model_config.get('mlflow', {})

    tracking_uri = os.environ.get(
        'MLFLOW_TRACKING_URI',
        mlflow_wf.get('tracking_uri') or mlflow_mdl.get('tracking_uri', 'http://localhost:5000'),
    )
    mlflow.set_tracking_uri(tracking_uri)

    registry_name     = mlflow_wf.get('registry_name', mlflow_mdl.get('registry_name', 'absa_indobert'))
    production_stage  = mlflow_wf.get('production_stage', 'Production')
    comparison_metric = workflow_config['model_validation']['comparison_metric']

    client = MlflowClient()
    try:
        versions = client.get_latest_versions(registry_name, stages=[production_stage])
    except MlflowException as exc:
        if 'RESOURCE_DOES_NOT_EXIST' in str(exc.error_code):
            return {
                'exists': False,
                'value' : None,
                'reason': (
                    f"Registered model '{registry_name}' belum ada di MLflow Model "
                    f"Registry (belum pernah ada eksperimen yang didaftarkan)."
                ),
            }
        return {
            'exists': False,
            'value' : None,
            'reason': f"Gagal mengakses MLflow Model Registry: {exc}",
        }
    except Exception as exc:
        return {
            'exists': False,
            'value' : None,
            'reason': f"Gagal mengakses MLflow Model Registry: {exc}",
        }

    if not versions:
        return {
            'exists': False,
            'value' : None,
            'reason': (
                f"Registered model '{registry_name}' ada, tetapi belum ada versi "
                f"berstage '{production_stage}'."
            ),
        }

    prod_run_id = versions[0].run_id
    try:
        prod_run     = client.get_run(prod_run_id)
        prod_metrics = prod_run.data.metrics
        return {
            'exists': True,
            'value' : prod_metrics.get(comparison_metric),
            'reason': f"Model produksi ditemukan: '{registry_name}' v{versions[0].version}",
        }
    except Exception as exc:
        return {
            'exists': False,
            'value' : None,
            'reason': f"Model produksi terdaftar tetapi gagal membaca run metrics: {exc}",
        }


def _check_vs_production(
    metrics: dict,
    production_lookup: dict,
    validation_cfg: dict,
) -> dict:
    """
    Bandingkan metrik model baru dengan model produksi.

    Jika belum ada model produksi (production_lookup['exists'] = False),
    keputusan ditentukan oleh model_validation.skip_comparison_if_no_production:
      True  → uji 2 lulus otomatis (mode bootstrap)
      False → uji 2 gagal, mewajibkan registrasi model produksi terlebih dahulu
    """
    comparison_metric   = validation_cfg['comparison_metric']
    require_improvement = validation_cfg.get('require_improvement', True)
    skip_if_no_prod      = validation_cfg.get('skip_comparison_if_no_production', True)

    new_value = metrics.get(comparison_metric, 0.0)

    if not production_lookup['exists']:
        if skip_if_no_prod:
            return {
                'passed'        : True,
                'reasons'       : [],
                'bootstrap_mode': True,
                'details': {
                    'production_model': None,
                    'new_value'       : new_value,
                    'prod_value'      : None,
                    'note'            : production_lookup['reason'],
                },
            }
        else:
            return {
                'passed'        : False,
                'reasons'       : [
                    f"{production_lookup['reason']} "
                    f"(skip_comparison_if_no_production=False — registrasi model "
                    f"produksi diwajibkan sebelum pipeline dapat lolos validasi)."
                ],
                'bootstrap_mode': True,
                'details': {
                    'production_model': None,
                    'new_value'       : new_value,
                    'prod_value'      : None,
                    'note'            : production_lookup['reason'],
                },
            }

    prod_metric_value = production_lookup['value']

    if require_improvement:
        passed = new_value > prod_metric_value
        reasons = (
            []
            if passed
            else [
                f"{comparison_metric} model baru ({new_value:.4f}) tidak lebih baik "
                f"dari produksi ({prod_metric_value:.4f})"
            ]
        )
    else:
        passed  = True
        reasons = []

    return {
        'passed'        : passed,
        'reasons'       : reasons,
        'bootstrap_mode': False,
        'details': {
            'metric'    : comparison_metric,
            'new_value' : new_value,
            'prod_value': prod_metric_value,
            'delta'     : new_value - prod_metric_value,
        },
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_validate_model(metrics: dict, workflow_config: dict, model_config: dict) -> dict:
    """
    Jalankan validasi model dua tahap.

    Parameters
    ----------
    metrics         : dict — output run_evaluate_model() (metrik test set)
    workflow_config : dict — konfigurasi workflow dari pipeline_config.yaml
    model_config    : dict — konfigurasi model dari YAML eksperimen

    Returns
    -------
    dict:
      passed              : bool — True jika kedua uji lulus
      threshold_check     : dict — hasil uji 1
      production_check    : dict — hasil uji 2
      failure_reasons     : list[str] — daftar alasan kegagalan (kosong jika lulus)
    """
    validation_cfg = workflow_config['model_validation']

    print("\n  [Uji 1] Memeriksa threshold metrik absolut...")
    threshold_result = _check_threshold(metrics, validation_cfg)
    status1 = 'LULUS' if threshold_result['passed'] else 'GAGAL'
    print(f"  Uji 1: {status1}")
    for reason in threshold_result['reasons']:
        print(f"    - {reason}")

    print("\n  [Uji 2] Membandingkan dengan model produksi...")
    production_lookup = _get_production_metric(workflow_config, model_config)
    production_result = _check_vs_production(metrics, production_lookup, validation_cfg)
    status2 = 'LULUS' if production_result['passed'] else 'GAGAL'

    if production_result['bootstrap_mode']:
        print(f"  [MODE BOOTSTRAP] {production_lookup['reason']}")

    prod_value = production_lookup['value']
    prod_label = f"{prod_value:.4f}" if prod_value is not None else "tidak ada"
    print(f"  Uji 2: {status2} (produksi: {prod_label}, "
          f"baru: {metrics.get(validation_cfg['comparison_metric'], 0):.4f})")
    for reason in production_result['reasons']:
        print(f"    - {reason}")

    all_failure_reasons = threshold_result['reasons'] + production_result['reasons']
    overall_passed = threshold_result['passed'] and production_result['passed']

    if overall_passed:
        print("\n  Model LOLOS validasi dan siap didaftarkan.")
    else:
        print(f"\n  Model TIDAK LOLOS validasi ({len(all_failure_reasons)} alasan).")

    return {
        'passed'           : overall_passed,
        'threshold_check'  : threshold_result,
        'production_check' : production_result,
        'failure_reasons'  : all_failure_reasons,
    }
