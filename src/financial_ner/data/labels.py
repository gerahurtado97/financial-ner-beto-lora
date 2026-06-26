"""
labels.py — Mapeo entre etiquetas BIO (string) e índices enteros.

Hugging Face Trainer trabaja con índices enteros, no strings.
Este módulo centraliza la conversión para que el orden sea consistente
en todo el proyecto — entrenamiento, evaluación e inferencia deben
usar EXACTAMENTE el mismo mapeo o las métricas serán incorrectas.
"""

from __future__ import annotations

# Orden fijo — debe coincidir con configs/model_config.yaml
LABEL_LIST = [
    "O",
    "B-INSTR", "I-INSTR",
    "B-MONTO", "I-MONTO",
    "B-PLAZO", "I-PLAZO",
    "B-TASA", "I-TASA",
    "B-ENTIDAD", "I-ENTIDAD",
]

LABEL_TO_ID: dict[str, int] = {label: idx for idx, label in enumerate(LABEL_LIST)}
ID_TO_LABEL: dict[int, str] = {idx: label for idx, label in enumerate(LABEL_LIST)}

# Índice especial de Hugging Face / PyTorch CrossEntropyLoss para ignorar
# un token en el cálculo de la pérdida (subtokens que no son el inicio
# de una palabra original)
IGNORE_INDEX = -100


def labels_to_ids(labels: list[str]) -> list[int]:
    """Convierte una lista de etiquetas string a índices enteros."""
    return [LABEL_TO_ID[label] for label in labels]


def ids_to_labels(ids: list[int]) -> list[str]:
    """
    Convierte una lista de índices a etiquetas string.

    Los índices IGNORE_INDEX (-100) se mapean a None — el caller debe
    filtrarlos antes de usar seqeval.
    """
    result = []
    for idx in ids:
        if idx == IGNORE_INDEX:
            result.append(None)
        else:
            result.append(ID_TO_LABEL[idx])
    return result
