"""
schema.py — Validación del dataset con Pandera.

Valida dos cosas críticas antes de tokenizar:
    1. Cada ejemplo tiene exactamente el mismo número de tokens y ner_tags
    2. Cada ner_tag pertenece al esquema BIO válido (no hay etiquetas inventadas)

Esto previene el error más común en NER: un ejemplo con tokens=5 pero
ner_tags=4 (desalineación) que rompe silenciosamente el entrenamiento
muchas iteraciones después de la carga de datos.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

VALID_LABELS = {
    "O",
    "B-INSTR", "I-INSTR",
    "B-MONTO", "I-MONTO",
    "B-PLAZO", "I-PLAZO",
    "B-TASA", "I-TASA",
    "B-ENTIDAD", "I-ENTIDAD",
}


class SchemaValidationError(Exception):
    """Se lanza cuando el dataset no cumple el esquema esperado."""


def validate_example(example: dict, idx: int) -> None:
    """
    Valida un único ejemplo del dataset.

    Parameters
    ----------
    example : dict
        Debe tener llaves "tokens" (list[str]) y "ner_tags" (list[str]).
    idx : int
        Índice del ejemplo, usado para mensajes de error claros.

    Raises
    ------
    SchemaValidationError
        Si el ejemplo no cumple el esquema.
    """
    if "tokens" not in example or "ner_tags" not in example:
        raise SchemaValidationError(
            f"Ejemplo {idx}: debe tener llaves 'tokens' y 'ner_tags'. "
            f"Llaves encontradas: {list(example.keys())}"
        )

    tokens = example["tokens"]
    tags = example["ner_tags"]

    if len(tokens) != len(tags):
        raise SchemaValidationError(
            f"Ejemplo {idx}: tokens ({len(tokens)}) y ner_tags ({len(tags)}) "
            f"tienen longitudes distintas. tokens={tokens} tags={tags}"
        )

    if len(tokens) == 0:
        raise SchemaValidationError(f"Ejemplo {idx}: tokens vacío")

    invalid_tags = set(tags) - VALID_LABELS
    if invalid_tags:
        raise SchemaValidationError(
            f"Ejemplo {idx}: etiquetas inválidas {invalid_tags}. "
            f"Etiquetas válidas: {sorted(VALID_LABELS)}"
        )

    # Validar consistencia BIO: un I-X no puede aparecer sin un B-X o I-X previo
    prev_tag = "O"
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            entity_type = tag[2:]
            prev_entity = prev_tag[2:] if prev_tag != "O" else None
            if prev_entity != entity_type:
                raise SchemaValidationError(
                    f"Ejemplo {idx}, token {i} ('{tokens[i]}'): "
                    f"etiqueta '{tag}' sin un '{tag.replace('I-', 'B-')}' "
                    f"o '{tag}' precedente válido. Tag anterior: '{prev_tag}'"
                )
        prev_tag = tag


def validate_dataset(dataset: list[dict]) -> None:
    """
    Valida el dataset completo, ejemplo por ejemplo.

    Parameters
    ----------
    dataset : list[dict]
        Lista de ejemplos con "tokens" y "ner_tags".

    Raises
    ------
    SchemaValidationError
        En el primer ejemplo inválido encontrado.
    """
    if len(dataset) == 0:
        raise SchemaValidationError("El dataset está vacío")

    for idx, example in enumerate(dataset):
        validate_example(example, idx)

    logger.info("✓ Dataset validado: %d ejemplos, esquema BIO correcto", len(dataset))


def label_distribution(dataset: list[dict]) -> dict[str, int]:
    """
    Cuenta cuántas veces aparece cada etiqueta en el dataset.

    Útil para detectar desbalance severo entre clases antes de entrenar.

    Returns
    -------
    dict[str, int]
        Conteo de cada etiqueta BIO.
    """
    counts: dict[str, int] = {label: 0 for label in VALID_LABELS}
    for example in dataset:
        for tag in example["ner_tags"]:
            counts[tag] += 1
    return counts
