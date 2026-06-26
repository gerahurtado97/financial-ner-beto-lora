"""
alignment.py — Alineación de etiquetas BIO con subtokens del tokenizador.

Esta es la pieza más crítica técnicamente del proyecto (señalada
explícitamente en el enunciado y en la Clase 6 del diplomado).

El problema:
    Nuestro dataset tiene etiquetas a nivel de PALABRA:
        ["CETES", "a", "91", "días"]  →  ["B-INSTR", "O", "B-PLAZO", "I-PLAZO"]

    Pero BETO usa WordPiece y puede partir una palabra en subtokens:
        "CETES" → ["CE", "##TES"]   (2 subtokens para 1 palabra)

    Si no realineamos, el modelo recibe más subtokens que etiquetas
    y el entrenamiento falla o, peor, falla SILENCIOSAMENTE con un
    desplazamiento de etiquetas que arruina el aprendizaje sin
    levantar ningún error.

La solución estándar (Hugging Face docs, Clase 6):
    1. Tokenizar con word_ids() para saber a qué palabra original
       pertenece cada subtoken
    2. El PRIMER subtoken de cada palabra recibe la etiqueta original
    3. Los subtokens SIGUIENTES de la misma palabra reciben IGNORE_INDEX (-100)
    4. Los tokens especiales ([CLS], [SEP], [PAD]) también reciben -100

Ejemplo completo:
    Palabra:    CETES        a   91     días
    Etiqueta:   B-INSTR      O   B-PLAZO I-PLAZO
    Subtokens:  [CLS] CE ##TES  a  91  días  [SEP]
    Etiq. final: -100  B-INSTR -100  O  B-PLAZO I-PLAZO  -100
"""

from __future__ import annotations

from transformers import BatchEncoding, PreTrainedTokenizerFast

from financial_ner.data.labels import IGNORE_INDEX, LABEL_TO_ID


def align_labels_with_tokens(
    tokens: list[str],
    ner_tags: list[str],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
) -> BatchEncoding:
    """
    Tokeniza una secuencia de palabras y realinea sus etiquetas a subtokens.

    Parameters
    ----------
    tokens : list[str]
        Palabras originales (ya separadas, como en el dataset JSON).
    ner_tags : list[str]
        Etiquetas BIO, una por palabra. Debe tener la misma longitud que tokens.
    tokenizer : PreTrainedTokenizerFast
        Tokenizador de BETO (debe ser "fast" para soportar word_ids()).
    max_length : int
        Longitud máxima de secuencia (con padding/truncamiento).

    Returns
    -------
    BatchEncoding
        Incluye input_ids, attention_mask, y la llave adicional "labels"
        con los índices de etiqueta alineados a nivel de subtoken,
        usando IGNORE_INDEX (-100) donde corresponda.

    Raises
    ------
    ValueError
        Si tokens y ner_tags no tienen la misma longitud.
    """
    if len(tokens) != len(ner_tags):
        raise ValueError(
            f"tokens ({len(tokens)}) y ner_tags ({len(ner_tags)}) "
            f"deben tener la misma longitud"
        )

    # is_split_into_words=True le dice al tokenizer que "tokens" ya son
    # palabras separadas — él decide cómo partirlas en subtokens
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    # word_ids() devuelve, para cada subtoken, el índice de la palabra
    # original a la que pertenece (o None para tokens especiales)
    word_ids = encoding.word_ids()

    aligned_labels: list[int] = []
    previous_word_id: int | None = None

    for word_id in word_ids:
        if word_id is None:
            # Token especial: [CLS], [SEP], [PAD]
            aligned_labels.append(IGNORE_INDEX)
        elif word_id != previous_word_id:
            # Primer subtoken de una palabra nueva: usa la etiqueta real
            aligned_labels.append(LABEL_TO_ID[ner_tags[word_id]])
        else:
            # Subtoken adicional de la MISMA palabra: ignorar en la loss
            aligned_labels.append(IGNORE_INDEX)
        previous_word_id = word_id

    encoding["labels"] = aligned_labels
    return encoding


def align_dataset(
    dataset: list[dict],
    tokenizer: PreTrainedTokenizerFast,
    max_length: int = 64,
) -> list[dict]:
    """
    Aplica align_labels_with_tokens a un dataset completo.

    Parameters
    ----------
    dataset : list[dict]
        Lista de {"tokens": [...], "ner_tags": [...]}.
    tokenizer : PreTrainedTokenizerFast
    max_length : int

    Returns
    -------
    list[dict]
        Lista de dicts con input_ids, attention_mask, labels — listos
        para alimentar al Trainer de Hugging Face.
    """
    aligned = []
    for example in dataset:
        encoding = align_labels_with_tokens(
            example["tokens"], example["ner_tags"], tokenizer, max_length
        )
        aligned.append({
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
            "labels": encoding["labels"],
        })
    return aligned


def count_subtoken_splits(
    dataset: list[dict],
    tokenizer: PreTrainedTokenizerFast,
) -> dict[str, int]:
    """
    Cuenta cuántas palabras del dataset se dividen en múltiples subtokens.

    Útil para el reporte — cuantifica qué tan seguido ocurre el problema
    de alineación que este módulo resuelve.

    Returns
    -------
    dict con "total_words", "split_words", "split_ratio"
    """
    total_words = 0
    split_words = 0

    for example in dataset:
        for word in example["tokens"]:
            total_words += 1
            n_subtokens = len(tokenizer.tokenize(word))
            if n_subtokens > 1:
                split_words += 1

    ratio = split_words / total_words if total_words > 0 else 0.0
    return {
        "total_words": total_words,
        "split_words": split_words,
        "split_ratio": round(ratio, 4),
    }
