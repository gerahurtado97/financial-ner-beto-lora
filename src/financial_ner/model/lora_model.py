"""
lora_model.py — Construcción del modelo BETO + LoRA para token classification.

LoRA (Low-Rank Adaptation, Hu et al. 2021):
    En lugar de actualizar la matriz de pesos completa W ∈ R^(d×d) de
    cada capa de atención, se aprende una actualización de rango bajo:

        W_ft = W_0 + B·A

    donde B ∈ R^(d×r), A ∈ R^(r×d), con r << d (rango bajo).
    Solo A y B son entrenables — W_0 (los pesos preentrenados de BETO)
    quedan congelados.

Por qué target_modules=["query", "value"]:
    Son las proyecciones W_Q y W_V de la self-attention (Clase 7).
    El paper original de LoRA encontró que adaptar Q y V es suficiente
    para la mayoría de tareas downstream — K se deja congelado.

Comparación r=8 vs r=16 (Clase 10 del diplomado):
    r mayor → más parámetros entrenables → más capacidad de adaptación
    pero también más riesgo de overfitting con un dataset pequeño (322 ejemplos)
"""

from __future__ import annotations

import logging

from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForTokenClassification, AutoTokenizer

from financial_ner.data.labels import ID_TO_LABEL, LABEL_LIST, LABEL_TO_ID

logger = logging.getLogger(__name__)


def load_base_model_and_tokenizer(
    base_model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
):
    """
    Carga BETO base y su tokenizador, configurado para token classification
    con las 11 etiquetas del esquema BIO del proyecto.

    Returns
    -------
    model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    model = AutoModelForTokenClassification.from_pretrained(
        base_model_name,
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        use_safetensors=True,
    )

    logger.info(
        "Modelo base cargado: %s (%s parámetros)",
        base_model_name,
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    return model, tokenizer


def build_lora_model(
    base_model_name: str = "dccuchile/bert-base-spanish-wwm-cased",
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> tuple[PeftModel, "AutoTokenizer"]:
    """
    Construye el modelo BETO con adaptadores LoRA para NER.

    Parameters
    ----------
    base_model_name : str
        Identificador de Hugging Face del modelo base.
    r : int
        Rango de las matrices LoRA. El proyecto compara r=8 vs r=16.
    lora_alpha : int
        Factor de escala — convención estándar es 2*r.
    lora_dropout : float
        Dropout aplicado en las capas LoRA.
    target_modules : list[str] | None
        Qué proyecciones lineales adaptar. Por defecto ["query", "value"]
        (Q y V de la self-attention, Clase 7).

    Returns
    -------
    model : PeftModel
        Modelo con adaptadores LoRA — solo A y B son entrenables.
    tokenizer : AutoTokenizer
    """
    if target_modules is None:
        target_modules = ["query", "value"]

    model, tokenizer = load_base_model_and_tokenizer(base_model_name)

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )

    lora_model = get_peft_model(model, lora_config)

    total = sum(p.numel() for p in lora_model.parameters())
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)

    logger.info(
        "LoRA configurado: r=%d, alpha=%d, target_modules=%s",
        r, lora_alpha, target_modules,
    )
    logger.info(
        "Parámetros entrenables: %s / %s (%.3f%%)",
        f"{trainable:,}", f"{total:,}", 100 * trainable / total,
    )

    return lora_model, tokenizer


def count_trainable_parameters(model: PeftModel) -> dict[str, int | float]:
    """
    Resumen de parámetros entrenables vs totales — para el reporte.

    Returns
    -------
    dict con "total", "trainable", "trainable_pct"
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_pct": round(100 * trainable / total, 4) if total > 0 else 0.0,
    }
