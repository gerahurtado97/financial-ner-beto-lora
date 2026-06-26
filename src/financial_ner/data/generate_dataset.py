"""
generate_dataset.py — Generador del dataset sintético de NER financiero.

Combina plantillas (templates.py) con vocabularios realistas
(vocabularies.py) para producir ejemplos etiquetados en BIO sin
necesidad de anotación manual.

CORRECCIÓN CRÍTICA — separación de vocabulario por split:
    La primera versión de este generador usaba el MISMO vocabulario
    para train y val/test, lo que permitía al modelo memorizar
    palabras ("CETES siempre es B-INSTR") y alcanzar F1=1.0 sin
    aprender el patrón BIO real. Resultado: eval_f1=1.0 desde la
    época 4 — señal inequívoca de overfitting, no de buen ajuste.

    La solución: vocabulary.py ahora expone pools TRAIN y HELDOUT
    disjuntos por categoría. generate_dataset() acepta un parámetro
    `vocab_pool` ("train" o "heldout") que selecciona qué subconjunto
    usar. train.json se genera con vocab_pool="train"; val.json y
    test.json se generan con vocab_pool="heldout" — el modelo nunca
    ve esas palabras exactas durante el entrenamiento.
"""

from __future__ import annotations

import logging
import random
from typing import Literal

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
    t_negativo_sin_entidades,
    t_simple_instr_entidad,
)
from financial_ner.data.vocabularies import (
    ENTIDADES_HELDOUT,
    ENTIDADES_TRAIN,
    INSTRUMENTOS_HELDOUT,
    INSTRUMENTOS_TRAIN,
    MONTOS_HELDOUT,
    MONTOS_TRAIN,
    PLAZOS_HELDOUT,
    PLAZOS_TRAIN,
    TASAS_HELDOUT,
    TASAS_TRAIN,
    VERBOS_POLITICA,
)

logger = logging.getLogger(__name__)

VocabPool = Literal["train", "heldout"]


def _get_pools(vocab_pool: VocabPool) -> dict[str, list[str]]:
    """Selecciona los pools de vocabulario correctos según el split."""
    if vocab_pool == "train":
        return {
            "instrumentos": INSTRUMENTOS_TRAIN,
            "entidades": ENTIDADES_TRAIN,
            "plazos": PLAZOS_TRAIN,
            "tasas": TASAS_TRAIN,
            "montos": MONTOS_TRAIN,
        }
    elif vocab_pool == "heldout":
        return {
            "instrumentos": INSTRUMENTOS_HELDOUT,
            "entidades": ENTIDADES_HELDOUT,
            "plazos": PLAZOS_HELDOUT,
            "tasas": TASAS_HELDOUT,
            "montos": MONTOS_HELDOUT,
        }
    raise ValueError(f"vocab_pool debe ser 'train' o 'heldout', recibido: {vocab_pool}")


def _sample(rng: random.Random, pool: list[str]) -> str:
    return rng.choice(pool)


def generate_dataset(
    n_per_template: int = 25,
    n_negative: int = 40,
    seed: int = 42,
    vocab_pool: VocabPool = "train",
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
    vocab_pool : "train" | "heldout"
        Qué subconjunto de vocabulario usar. "train" para generar el
        conjunto de entrenamiento; "heldout" para val/test, garantizando
        que el modelo nunca vio esas palabras exactas en entrenamiento.

    Returns
    -------
    list[dict]
        Lista de ejemplos {"tokens": [...], "ner_tags": [...]}.
    """
    pools = _get_pools(vocab_pool)
    rng = random.Random(seed)
    dataset: list[dict] = []

    # ── Plantilla 1: instrumento + plazo (sin tasa)
    for _ in range(n_per_template):
        instr = _sample(rng, pools["instrumentos"])
        plazo = _sample(rng, pools["plazos"])
        tokens, tags = t_instrumento_plazo(instr, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 2: instrumento + plazo + tasa
    for _ in range(n_per_template):
        instr = _sample(rng, pools["instrumentos"])
        plazo = _sample(rng, pools["plazos"])
        tasa = _sample(rng, pools["tasas"])
        tokens, tags = t_instrumento_plazo_tasa(instr, plazo, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 3: entidad + verbo + tasa
    for _ in range(n_per_template):
        entidad = _sample(rng, pools["entidades"])
        verbo = _sample(rng, VERBOS_POLITICA)
        tasa = _sample(rng, pools["tasas"])
        tokens, tags = t_entidad_verbo_tasa(entidad, verbo, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 4: entidad + verbo + instrumento + plazo
    for _ in range(n_per_template):
        entidad = _sample(rng, pools["entidades"])
        verbo = _sample(rng, VERBOS_POLITICA)
        instr = _sample(rng, pools["instrumentos"])
        plazo = _sample(rng, pools["plazos"])
        tokens, tags = t_entidad_verbo_instr_plazo(entidad, verbo, instr, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 5: entidad + monto + instrumento
    for _ in range(n_per_template):
        entidad = _sample(rng, pools["entidades"])
        monto = _sample(rng, pools["montos"])
        instr = _sample(rng, pools["instrumentos"])
        tokens, tags = t_entidad_emitio_monto(entidad, monto, instr)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 6: instrumento + plazo + entidad
    for _ in range(n_per_template):
        instr = _sample(rng, pools["instrumentos"])
        plazo = _sample(rng, pools["plazos"])
        entidad = _sample(rng, pools["entidades"])
        tokens, tags = t_instr_plazo_entidad(instr, plazo, entidad)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 7: monto + instrumento + tasa
    for _ in range(n_per_template):
        monto = _sample(rng, pools["montos"])
        instr = _sample(rng, pools["instrumentos"])
        tasa = _sample(rng, pools["tasas"])
        tokens, tags = t_monto_instr_tasa(monto, instr, tasa)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 8: entidad + tasa + plazo (oración larga, 3 entidades)
    for _ in range(n_per_template):
        entidad = _sample(rng, pools["entidades"])
        tasa = _sample(rng, pools["tasas"])
        plazo = _sample(rng, pools["plazos"])
        tokens, tags = t_entidad_recorte_tasa_plazo(entidad, tasa, plazo)
        dataset.append({"tokens": tokens, "ner_tags": tags})

    # ── Plantilla 9: instrumento + entidad (oración corta)
    for _ in range(n_per_template):
        instr = _sample(rng, pools["instrumentos"])
        entidad = _sample(rng, pools["entidades"])
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
        "Dataset generado (vocab_pool=%s): %d ejemplos (%d positivos + %d negativos)",
        vocab_pool, len(dataset), len(dataset) - n_negative, n_negative,
    )

    return dataset


def generate_train_val_test(
    n_per_template_train: int = 35,
    n_negative_train: int = 50,
    n_per_template_evalsplit: int = 8,
    n_negative_evalsplit: int = 12,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Genera train (vocab_pool="train") y val+test (vocab_pool="heldout")
    en una sola llamada, garantizando vocabulario disjunto entre ambos.

    val y test usan semillas distintas para no producir exactamente
    las mismas oraciones entre sí, aunque comparten el pool heldout.

    Parameters
    ----------
    n_per_template_train : int
        Ejemplos por plantilla para el set de entrenamiento.
    n_negative_train : int
        Ejemplos negativos para entrenamiento.
    n_per_template_evalsplit : int
        Ejemplos por plantilla para val y test (cada uno).
    n_negative_evalsplit : int
        Ejemplos negativos para val y test (cada uno).
    seed : int

    Returns
    -------
    train, val, test : list[dict]
    """
    train = generate_dataset(
        n_per_template=n_per_template_train,
        n_negative=n_negative_train,
        seed=seed,
        vocab_pool="train",
    )
    val = generate_dataset(
        n_per_template=n_per_template_evalsplit,
        n_negative=n_negative_evalsplit,
        seed=seed + 1,
        vocab_pool="heldout",
    )
    test = generate_dataset(
        n_per_template=n_per_template_evalsplit,
        n_negative=n_negative_evalsplit,
        seed=seed + 2,
        vocab_pool="heldout",
    )

    train = deduplicate(train)
    val = deduplicate(val)
    test = deduplicate(test)

    # Verificación defensiva: val y test no deben compartir oraciones exactas
    val_keys = {tuple(ex["tokens"]) for ex in val}
    test_keys = {tuple(ex["tokens"]) for ex in test}
    overlap = val_keys & test_keys
    if overlap:
        logger.warning(
            "%d oraciones idénticas entre val y test (mismo vocab_pool, "
            "puede ocurrir por azar) — no afecta la separación train/heldout",
            len(overlap),
        )

    return train, val, test


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
