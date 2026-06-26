"""
metrics.py — Métricas de evaluación con seqeval (nivel de ENTIDAD, no de token).

Por qué F1 a nivel de entidad y no de token (Clase 11):
    Una entidad de 3 tokens como "100 millones de pesos" debe contarse
    como UN acierto o UN error — no tres aciertos parciales por token.
    seqeval implementa exactamente esta semántica — es el estándar de
    la industria para NER.
"""

from __future__ import annotations

import numpy as np
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from financial_ner.data.labels import ID_TO_LABEL, IGNORE_INDEX


def predictions_to_labels(
    predictions: np.ndarray,
    label_ids: np.ndarray,
) -> tuple[list[list[str]], list[list[str]]]:
    """
    Convierte logits/predicciones e ids de etiqueta a listas de strings BIO,
    filtrando las posiciones con IGNORE_INDEX (subtokens y padding).

    Parameters
    ----------
    predictions : np.ndarray shape (n_examples, seq_len, n_labels) o (n_examples, seq_len)
    label_ids : np.ndarray shape (n_examples, seq_len)

    Returns
    -------
    true_labels, pred_labels : list[list[str]]
    """
    if predictions.ndim == 3:
        pred_ids = np.argmax(predictions, axis=-1)
    else:
        pred_ids = predictions

    true_labels: list[list[str]] = []
    pred_labels: list[list[str]] = []

    for true_seq, pred_seq in zip(label_ids, pred_ids):
        true_seq_labels = []
        pred_seq_labels = []
        for true_id, pred_id in zip(true_seq, pred_seq):
            if true_id == IGNORE_INDEX:
                continue
            true_seq_labels.append(ID_TO_LABEL[int(true_id)])
            pred_seq_labels.append(ID_TO_LABEL[int(pred_id)])
        true_labels.append(true_seq_labels)
        pred_labels.append(pred_seq_labels)

    return true_labels, pred_labels


def compute_metrics(eval_pred) -> dict[str, float]:
    """
    Función compute_metrics para el Trainer de Hugging Face.

    Returns
    -------
    dict con "precision", "recall", "f1".
    """
    predictions, label_ids = eval_pred
    true_labels, pred_labels = predictions_to_labels(predictions, label_ids)

    return {
        "precision": precision_score(true_labels, pred_labels),
        "recall": recall_score(true_labels, pred_labels),
        "f1": f1_score(true_labels, pred_labels),
    }


def full_classification_report(
    predictions: np.ndarray,
    label_ids: np.ndarray,
) -> str:
    """Reporte detallado de precision/recall/f1 POR TIPO DE ENTIDAD."""
    true_labels, pred_labels = predictions_to_labels(predictions, label_ids)
    return classification_report(true_labels, pred_labels, digits=4)


def per_entity_scores(
    predictions: np.ndarray,
    label_ids: np.ndarray,
) -> dict[str, dict[str, float]]:
    """
    Métricas P/R/F1 desglosadas por tipo de entidad, en formato dict.

    Returns
    -------
    dict[str, dict] — ej: {"INSTR": {"precision": 0.9, "recall": 0.85, "f1": 0.87}, ...}
    """
    true_labels, pred_labels = predictions_to_labels(predictions, label_ids)

    entity_types = ["INSTR", "MONTO", "PLAZO", "TASA", "ENTIDAD"]
    scores: dict[str, dict[str, float]] = {}

    for entity_type in entity_types:
        def filter_type(labels_seq: list[str], _et: str = entity_type) -> list[str]:
            return [
                label if label == "O" or _et in label else "O"
                for label in labels_seq
            ]

        true_filtered = [filter_type(seq) for seq in true_labels]
        pred_filtered = [filter_type(seq) for seq in pred_labels]

        scores[entity_type] = {
            "precision": precision_score(true_filtered, pred_filtered),
            "recall": recall_score(true_filtered, pred_filtered),
            "f1": f1_score(true_filtered, pred_filtered),
        }

    return scores
