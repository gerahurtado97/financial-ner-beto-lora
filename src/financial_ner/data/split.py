"""
split.py — División del dataset en train/val/test.

A diferencia del Titanic (clasificación binaria estratificada por target),
en NER no hay una sola etiqueta por ejemplo sino una secuencia de
etiquetas por token. Usamos split aleatorio simple — con un dataset
generado por plantillas y vocabulario balanceado, el split aleatorio
ya preserva razonablemente bien la distribución de entidades.
"""

from __future__ import annotations

import random


def train_val_test_split(
    dataset: list[dict],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Divide el dataset en tres splits aleatorios.

    Parameters
    ----------
    dataset : list[dict]
        Dataset completo de ejemplos {"tokens": ..., "ner_tags": ...}.
    train_size, val_size, test_size : float
        Proporciones que deben sumar 1.0 (con tolerancia de redondeo).
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    train, val, test : list[dict]
    """
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Las proporciones deben sumar 1.0, suman {total}")

    rng = random.Random(seed)
    shuffled = dataset.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]

    return train, val, test
