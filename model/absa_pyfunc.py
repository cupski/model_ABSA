"""
Wrapper Pyfunc untuk ABSAModel
================================
Membungkus bundle artifact (best_model.pt + tokenizer/ + config.yaml) hasil
pipeline/train_model.py menjadi flavor mlflow.pyfunc, sehingga model dapat
dimuat lewat satu entrypoint standar `mlflow.pyfunc.load_model()` — baik oleh
MLflow Model Registry, `mlflow models serve`, maupun aplikasi konsumen.

Hanya wrapper tipis ini yang di-pickle MLflow saat log_model(); bobot model
tetap tersimpan sebagai file checkpoint biasa dan dimuat ulang lewat
load_state_dict() di load_context() — bukan lewat pickle penuh objek
ABSAModel/encoder, sehingga tidak rapuh terhadap perpindahan path modul
Python seperti yang coba dihindari register_model.py.
"""

import os

import mlflow.pyfunc


class ABSAPyfuncModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        import torch
        from transformers import AutoTokenizer

        from model.absa_model import ABSAModel
        from preprocessing.preprocessing_functions import (
            FINAL_ASPECTS, NUM_CLASSES, LABEL_NAMES, clean_text,
        )

        checkpoint_dir = context.artifacts['checkpoint']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ckpt = torch.load(
            os.path.join(checkpoint_dir, 'best_model.pt'),
            map_location=self.device, weights_only=False,
        )
        cfg     = ckpt['config']
        rep_cfg = cfg['representation']

        self.model = ABSAModel(
            model_name   = rep_cfg['model_name'],
            aspects      = FINAL_ASPECTS,
            num_classes  = NUM_CLASSES,
            dropout_rate = cfg['model']['params']['dropout_rate'],
        )
        self.model.load_state_dict(ckpt['model_state'])
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer   = AutoTokenizer.from_pretrained(checkpoint_dir)
        self.max_length  = rep_cfg['max_length']
        self.aspects     = FINAL_ASPECTS
        self.label_names = LABEL_NAMES
        self._clean_text = clean_text

    def predict(self, context, model_input, params=None):
        import torch
        import pandas as pd

        if isinstance(model_input, pd.DataFrame):
            texts = model_input.iloc[:, 0].tolist()
        elif isinstance(model_input, (list, tuple)):
            texts = list(model_input)
        else:
            texts = [model_input]

        cleaned  = [self._clean_text(t) for t in texts]
        encoding = self.tokenizer(
            cleaned, max_length=self.max_length, truncation=True,
            padding=True, return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(encoding['input_ids'], encoding['attention_mask'])

        results = []
        for i in range(len(texts)):
            pred = {
                asp: self.label_names[asp][torch.argmax(logits[asp][i]).item()]
                for asp in self.aspects
            }
            results.append(pred)

        return results
