"""
generate_dataset.py — Generador del dataset sintético de NER financiero.

Combina plantillas (templates.py) con vocabularios realistas
(vocabularies.py) para producir ejemplos etiquetados en BIO sin
necesidad de anotación manual.

Estrategia anti-overfitting:
    1. Variedad de plantillas (9 patrones sintácticos distintos)
    2. Variedad de vocabulario (MX + internacional, ~80 valores por categoría)
    3. Oraciones negativas (sin entidades) para no sobre-predecir
    4. Muestreo aleatorio con seed fijo — combinaciones no repetidas
"""

from __future__ import annotations

import logging
import random

from financial_ner.data.templates import (
    FRASES_NEGATIVAS,
    t_entidad_emitio_monto,
    t_entidad_recorte_tasa_plazo,
    t_entidad_verbo_instr_plazo,
    t_entidad_verbo_tasa,
    t_instr_plazo_entidad,
    t_instrumento_plazo,
    t_instrumento_plazo_tasa,
    t_monto_instr_tasa,
    t_simple_instr_entidad,
    t_negativo_sin_entidades,
)
from financial_ner.data.vocabularies import (
    ENTIDADES,
    INSTRUMENTOS,
    MONTOS,
    PLAZOS,
    TASAS,
    VERBOS_POLITICA,
)

logger = logging.getLogger(__name__)


def _sample(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def generate_dataset(
    n_per_template: int = 25,
    n_negative: int = 40,
    seed: int = 42,
) -> list[dict]:
    """
    Genera el dataset sintético completo.

    Parameters
    ----------
    n_per_template : int
        Número de ejemplos a generar por cada una de las 9 plantillas
        positivas (con entidades).
    n_negative : int
        Número de ejemplos negativos (sin entidades) a incluir.
    seed : int
        Semilla para reproducibilidad.

    Returns
    -------
    list[dict]
        Lista de ejemplos {"tokens": [...], "ner_tags": [...]}.

    Notes
    -----
    Con n_per_template=25 y 9 plantillas + 40 negativos:
        9 * 25 + 40 = 265 ejemplos totales

    El tamaño se eligió deliberadamente moderado — el enunciado del
    proyecto indica que no hay mínimo requerido y que un dataset
    pequeño bien construido es preferible a uno grande con ruido.
    Suficiente variedad para LoRA con r=8/16 sin sobreajustar.
    """
    rng = random.Random(seed)
    dataset: list[dict] = []

    # ── Plantilla 1: instrumento + plazo (sin tasa)
    for _ in range(n_per_template):
        instr = _sample(rng, INSTRUMENTOS)
        plazo = _sample(rng, PLAZOS)
        tokens, tags = t_instrumento_plazo(instr, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 2: instrumento + plazo + tasa
    for _ in range(n_per_template):
        instr = _sample(rng, INSTRUMENTOS)
        plazo = _sample(rng, PLAZOS)
        tasa = _sample(rng, TASAS)
        tokens, tags = t_instrumento_plazo_tasa(instr, plazo, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 3: entidad + verbo + tasa
    for _ in range(n_per_template):
        entidad = _sample(rng, ENTIDADES)
        verbo = _sample(rng, VERBOS_POLITICA)
        tasa = _sample(rng, TASAS)
        tokens, tags = t_entidad_verbo_tasa(entidad, verbo, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 4: entidad + verbo + instrumento + plazo
    for _ in range(n_per_template):
        entidad = _sample(rng, ENTIDADES)
        verbo = _sample(rng, VERBOS_POLITICA)
        instr = _sample(rng, INSTRUMENTOS)
        plazo = _sample(rng, PLAZOS)
        tokens, tags = t_entidad_verbo_instr_plazo(entidad, verbo, instr, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 5: entidad + monto + instrumento
    for _ in range(n_per_template):
        entidad = _sample(rng, ENTIDADES)
        monto = _sample(rng, MONTOS)
        instr = _sample(rng, INSTRUMENTOS)
        tokens, tags = t_entidad_emitio_monto(entidad, monto, instr)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 6: instrumento + plazo + entidad
    for _ in range(n_per_template):
        instr = _sample(rng, INSTRUMENTOS)
        plazo = _sample(rng, PLAZOS)
        entidad = _sample(rng, ENTIDADES)
        tokens, tags = t_instr_plazo_entidad(instr, plazo, entidad)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 7: monto + instrumento + tasa
    for _ in range(n_per_template):
        monto = _sample(rng, MONTOS)
        instr = _sample(rng, INSTRUMENTOS)
        tasa = _sample(rng, TASAS)
        tokens, tags = t_monto_instr_tasa(monto, instr, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 8: entidad + tasa + plazo (oración larga, 4 entidades)
    for _ in range(n_per_template):
        entidad = _sample(rng, ENTIDADES)
        tasa = _sample(rng, TASAS)
        plazo = _sample(rng, PLAZOS)
        tokens, tags = t_entidad_recorte_tasa_plazo(entidad, tasa, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 9: instrumento + entidad (oración corta)
    for _ in range(n_per_template):
        instr = _sample(rng, INSTRUMENTOS)
        entidad = _sample(rng, ENTIDADES)
        tokens, tags = t_simple_instr_entidad(instr, entidad)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Ejemplos negativos: sin entidades financieras
    negativos_disponibles = FRASES_NEGATIVAS * ((n_negative // len(FRASES_NEGATIVAS)) + 1)
    rng.shuffle(negativos_disponibles)
    for frase in negativos_disponibles[:n_negative]:
        tokens, tags = t_negativo_sin_entidades(frase)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # Shuffle final para que el dataset no quede agrupado por plantilla
    rng.shuffle(dataset)

    logger.info(
        "Dataset generado: %d ejemplos (%d positivos + %d negativos)",
        len(dataset), len(dataset) - n_negative, n_negative,
    )

    return dataset


def deduplicate(dataset: list[dict]) -> list[dict]:
    """
    Elimina ejemplos duplicados (misma secuencia de tokens exacta).

    Con muestreo aleatorio sobre vocabularios finitos pueden generarse
    combinaciones idénticas — esto las filtra antes de hacer el split.
    """
    seen = set()
    unique = []
    for example in dataset:
        key = tuple(example["tokens"])
        if key not in seen:
            seen.add(key)
            unique.append(example)
    n_removed = len(dataset) - len(unique)
    if n_removed > 0:
        logger.info("Eliminados %d ejemplos duplicados", n_removed)
    return unique
