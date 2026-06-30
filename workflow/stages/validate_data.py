"""
Tahap Validasi Data (Workflow Wrapper)
======================================
Membungkus pipeline/validate_data.py menjadi satu unit eksekusi yang dapat
diisolasi dan diulang secara independen dalam automated ML workflow pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.validate_data import validate_data


def run_validate_data(model_config: dict) -> dict:
    """
    Validasi integritas dataset sebelum proses pelatihan.

    Menghentikan pipeline (raise RuntimeError) jika validasi gagal,
    sehingga step berikutnya tidak dieksekusi.

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen

    Returns
    -------
    dict — laporan validasi (total_rows, issues, passed)
    """
    report = validate_data(model_config)

    print(f"  Total baris dataset: {report['total_rows']}")
    if report['issues']:
        for issue in report['issues']:
            print(f"  [!] {issue}")

    if not report['passed']:
        raise RuntimeError(
            f"Validasi data gagal dengan {len(report['issues'])} masalah kritis: "
            f"{report['issues']}"
        )

    print("  Validasi lulus — semua pemeriksaan berhasil.")
    return report
