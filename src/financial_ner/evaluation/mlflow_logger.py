"""
mlflow_logger.py — Tracking de experimentos LoRA r=8 vs r=16 con MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow

logger = logging.getLogger(__name__)


def setup_mlflow(tracking_uri: str, experiment_name: str) -> str:
    """Configura MLflow y retorna el experiment_id."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    logger.info("MLflow configurado: experiment=%s id=%s", experiment_name, exp.experiment_id)
    return exp.experiment_id


def log_lora_run(
    lora_r: int,
    lora_alpha: int,
    param_counts: dict,
    train_history: list[dict],
    final_metrics: dict,
    per_entity_metrics: dict,
    model_path: Path | None = None,
    run_name: str | None = None,
) -> str:
    """
    Registra un run de fine-tuning LoRA en MLflow.

    Parameters
    ----------
    lora_r, lora_alpha : int
        Hiperparámetros LoRA de este run.
    param_counts : dict
        Salida de count_trainable_parameters() — total/trainable/trainable_pct.
    train_history : list[dict]
        Lista de logs por época del Trainer (loss, eval_f1, etc.).
    final_metrics : dict
        Métricas finales en test: precision, recall, f1.
    per_entity_metrics : dict
        Métricas desglosadas por tipo de entidad.
    model_path : Path | None
        Ruta a los pesos LoRA guardados.
    run_name : str | None

    Returns
    -------
    str — run_id de MLflow.
    """
    run_name = run_name or f"beto_lora_r{lora_r}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "base_model": "dccuchile/bert-base-spanish-wwm-cased",
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "target_modules": "query,value",
            "total_params": param_counts["total"],
            "trainable_params": param_counts["trainable"],
            "trainable_pct": param_counts["trainable_pct"],
        })

        for epoch_log in train_history:
            step = epoch_log.get("epoch", 0)
            metrics_to_log = {
                k: v for k, v in epoch_log.items()
                if isinstance(v, (int, float)) and k != "epoch"
            }
            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log, step=int(step * 100))

        mlflow.log_metrics({
            "test_precision": final_metrics.get("precision", 0),
            "test_recall": final_metrics.get("recall", 0),
            "test_f1": final_metrics.get("f1", 0),
        })

        for entity_type, scores in per_entity_metrics.items():
            for metric_name, value in scores.items():
                mlflow.log_metric(f"{entity_type}_{metric_name}", value)

        mlflow.set_tags({
            "model_type": "BETO + LoRA",
            "task": "token-classification",
            "scheme": "BIO",
            "dataset": "financial_ner_dataset (synthetic)",
        })

        if model_path is not None and Path(model_path).exists():
            mlflow.log_artifacts(str(model_path))

        run_id = run.info.run_id
        logger.info(
            "MLflow run registrado: %s (r=%d, F1=%.4f)",
            run_id, lora_r, final_metrics.get("f1", 0),
        )

    return run_id
