"""
Tahap Persiapan Data (Workflow Wrapper)
=========================================
Membungkus pipeline/prepare_data.py menjadi satu unit eksekusi yang dapat
diisolasi dan diulang secara independen dalam automated ML workflow pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pipeline.prepare_data import prepare_data


def run_prepare_data(model_config: dict) -> dict:
    """
    Terapkan prapemrosesan, bagi dataset menjadi split train/val/test,
    dan hitung class weights dari training set.

    Parameters
    ----------
    model_config : dict — konfigurasi model dari YAML eksperimen

    Returns
    -------
    dict — {'df_train', 'df_val', 'df_test', 'class_weights'}
    """
    data = prepare_data(model_config)

    n_train = len(data['df_train'])
    n_val   = len(data['df_val'])
    n_test  = len(data['df_test'])
    print(f"  Split dataset → Train: {n_train} | Val: {n_val} | Test: {n_test}")

    return data
