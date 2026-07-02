"""
Rekonstruksi ABSAModel dari Bundle Checkpoint
================================================
Satu fungsi bersama untuk memuat ulang model + tokenizer dari bundle
checkpoint (best_model.pt + tokenizer/) hasil pipeline/train_model.py.

Dipakai oleh dua konsumen:
  - workflow/stages/evaluate_model.py — evaluasi model yang baru dilatih
  - workflow/stages/validate_model.py — re-evaluasi model baseline yang
    diunduh dari MLflow Model Registry, untuk perbandingan Uji 3
"""

import os

import torch
from transformers import AutoTokenizer

from model.absa_model import ABSAModel
from preprocessing.preprocessing_functions import FINAL_ASPECTS, NUM_CLASSES


def load_model_from_checkpoint(save_dir: str, device: torch.device) -> tuple:
    """
    Rekonstruksi model dan tokenizer dari checkpoint yang disimpan train_model.

    best_model.pt berisi: epoch, model_state, val_sent_f1, val_det_f1,
    sent_f1_per_asp, det_f1_per_asp, config — sehingga rekonstruksi penuh
    tidak membutuhkan informasi tambahan di luar save_dir.

    Returns
    -------
    (model, tokenizer, config_from_ckpt)
    """
    ckpt_path = os.path.join(save_dir, 'best_model.pt')
    ckpt      = torch.load(ckpt_path, map_location=device, weights_only=False)

    cfg = ckpt['config']
    rep_cfg = cfg['representation']

    model = ABSAModel(
        model_name   = rep_cfg['model_name'],
        aspects      = FINAL_ASPECTS,
        num_classes  = NUM_CLASSES,
        dropout_rate = cfg['model']['params']['dropout_rate'],
    )
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()

    # Tokenizer dimuat dari save_dir (disimpan saat training via
    # tokenizer.save_pretrained) agar tidak bergantung pada akses HF Hub
    # maupun pergeseran versi model_name. Fallback ke Hub hanya untuk
    # checkpoint lama yang belum menyimpan tokenizer secara lokal.
    if os.path.isfile(os.path.join(save_dir, 'tokenizer_config.json')):
        tokenizer = AutoTokenizer.from_pretrained(save_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            rep_cfg['model_name'],
            revision=rep_cfg.get('model_revision', 'main'),
        )

    return model, tokenizer, cfg
