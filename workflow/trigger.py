"""
Entry Point Trigger Eksternal — ABSA Retraining Pipeline
==========================================================
Menyediakan dua mekanisme pemicu selain penjadwalan otomatis:

  1. Manual / ad-hoc via CLI:
       python workflow/trigger.py
       python workflow/trigger.py --config_path configs/experiment_indobert_baseline.yaml
       python workflow/trigger.py --reason "manual_retrain"

  2. Eksternal via HTTP (dari komponen monitoring saat mendeteksi drift/degradasi):
       python workflow/trigger.py --serve
       # Lalu POST http://localhost:8002/trigger

     Payload POST (opsional, semua memiliki default):
       {
         "config_path": "configs/experiment_indobert_baseline.yaml",
         "workflow_config_path": "workflow/pipeline_config.yaml",
         "reason": "monitoring_alert"
       }

Dengan adanya satu titik masuk ini, satu pipeline yang sama dapat dipicu
oleh tiga mekanisme (terjadwal, manual, monitoring) tanpa duplikasi
implementasi pipeline itu sendiri.
"""

import os
import sys
import subprocess
import argparse

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_FLOW_PATH = os.path.join(os.path.dirname(__file__), 'flow.py')


# ── Fungsi Pemicu Inti ────────────────────────────────────────────────────────

def trigger_pipeline(
    config_path: str           = 'configs/experiment_indobert_baseline.yaml',
    workflow_config_path: str  = 'workflow/pipeline_config.yaml',
    reason: str                = 'manual',
    wait: bool                 = False,
) -> dict:
    """
    Picu eksekusi ABSA Retraining Pipeline.

    Parameters
    ----------
    config_path          : path konfigurasi model (relatif terhadap root repo)
    workflow_config_path : path konfigurasi workflow
    reason               : 'scheduled' | 'manual' | 'monitoring_alert'
    wait                 : True = tunggu pipeline selesai; False = fire-and-forget

    Returns
    -------
    dict dengan kunci: status, pid, reason
    """
    cmd = [
        sys.executable, _FLOW_PATH, 'run',
        '--config_path',          config_path,
        '--workflow_config_path', workflow_config_path,
        '--trigger_reason',       reason,
    ]

    print(f"Memicu ABSA Retraining Pipeline...")
    print(f"  Alasan  : {reason}")
    print(f"  Config  : {config_path}")
    print(f"  Perintah: {' '.join(cmd)}")

    if wait:
        result = subprocess.run(cmd, cwd=_REPO_ROOT)
        return {
            'status'     : 'completed' if result.returncode == 0 else 'failed',
            'returncode' : result.returncode,
            'reason'     : reason,
        }
    else:
        proc = subprocess.Popen(cmd, cwd=_REPO_ROOT)
        print(f"  Pipeline berjalan di background (PID: {proc.pid})")
        return {
            'status': 'triggered',
            'pid'   : proc.pid,
            'reason': reason,
        }


# ── HTTP Server (untuk trigger eksternal dari komponen monitoring) ─────────────

def create_app():
    """Buat FastAPI app untuk menerima trigger dari komponen eksternal."""
    from fastapi import FastAPI
    from pydantic import BaseModel

    app = FastAPI(
        title       = 'ABSA Pipeline Trigger API',
        description = 'Endpoint untuk memicu ABSA Retraining Pipeline secara eksternal',
        version     = '1.0.0',
    )

    class TriggerRequest(BaseModel):
        config_path          : str = 'configs/experiment_indobert_baseline.yaml'
        workflow_config_path : str = 'workflow/pipeline_config.yaml'
        reason               : str = 'monitoring_alert'

    class TriggerResponse(BaseModel):
        status : str
        pid    : int | None = None
        reason : str
        message: str

    @app.get('/health')
    def health_check():
        """Periksa apakah trigger service berjalan."""
        return {'status': 'ok', 'service': 'ABSA Pipeline Trigger'}

    @app.post('/trigger', response_model=TriggerResponse)
    def trigger(request: TriggerRequest):
        """
        Picu eksekusi ABSA Retraining Pipeline.

        Dipanggil oleh komponen monitoring ketika mendeteksi penurunan
        performa model di produksi (model drift / performance degradation).
        """
        result = trigger_pipeline(
            config_path          = request.config_path,
            workflow_config_path = request.workflow_config_path,
            reason               = request.reason,
            wait                 = False,
        )
        return TriggerResponse(
            status  = result['status'],
            pid     = result.get('pid'),
            reason  = result['reason'],
            message = f"Pipeline dipicu dengan alasan: {result['reason']} (PID: {result.get('pid')})",
        )

    return app


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pemicu ABSA Retraining Pipeline (manual atau HTTP server)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  # Trigger manual langsung (tunggu selesai):
  python workflow/trigger.py --wait

  # Trigger manual di background:
  python workflow/trigger.py --reason "manual_retrain"

  # Jalankan HTTP server untuk trigger eksternal:
  python workflow/trigger.py --serve --port 8002

  # Trigger dari monitoring (kirim POST request):
  curl -X POST http://localhost:8002/trigger \\
       -H "Content-Type: application/json" \\
       -d '{"reason": "monitoring_alert"}'
        """,
    )
    parser.add_argument(
        '--config_path',
        default='configs/experiment_indobert_baseline.yaml',
        help='Path konfigurasi model YAML (relatif terhadap root repo)',
    )
    parser.add_argument(
        '--workflow_config_path',
        default='workflow/pipeline_config.yaml',
        help='Path konfigurasi workflow YAML',
    )
    parser.add_argument(
        '--reason',
        default='manual',
        choices=['manual', 'monitoring_alert', 'scheduled'],
        help='Alasan pemicu eksekusi pipeline',
    )
    parser.add_argument(
        '--wait',
        action='store_true',
        help='Tunggu pipeline selesai (blocking). Default: fire-and-forget.',
    )
    parser.add_argument(
        '--serve',
        action='store_true',
        help='Jalankan HTTP server untuk menerima trigger eksternal',
    )
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host HTTP server (default: 0.0.0.0)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8002,
        help='Port HTTP server (default: 8002)',
    )
    args = parser.parse_args()

    if args.serve:
        import uvicorn
        app = create_app()
        print(f"Memulai ABSA Pipeline Trigger API di http://{args.host}:{args.port}")
        print(f"  Dokumentasi API : http://localhost:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        result = trigger_pipeline(
            config_path          = args.config_path,
            workflow_config_path = args.workflow_config_path,
            reason               = args.reason,
            wait                 = args.wait,
        )
        print(f"\nHasil: {result}")
