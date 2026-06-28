import os
import mlflow

from run_experiment import load_config

config = load_config(config_path)
save_dir = config['model']['save_dir']

# Jika sudah ada .dvc pointer untuk model (setelah dvc add di laptop), log juga
model_dvc_candidates = [
    f for f in os.listdir(save_dir)
    if f.endswith('.dvc')
] if os.path.isdir(save_dir) else []
for dvc_file in model_dvc_candidates:
    mlflow.log_artifact(os.path.join(save_dir, dvc_file), artifact_path='model_ref')