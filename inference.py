#!/usr/bin/env python3
"""
inference.py — Script de inferencia para NER Financiero (entregable principal).

Carga el modelo BETO+LoRA fine-tuneado y extrae entidades financieras
de un archivo de texto en español.

Uso::

    python inference.py ruta/al/archivo.txt

El archivo .txt debe contener texto financiero en español (una o
varias oraciones, uno o varios párrafos). El script imprime cada
entidad encontrada con su tipo, una por línea:

    CETES INSTR
    91 días PLAZO
    11.25 por ciento TASA
    Banxico ENTIDAD

Robustez:
    - Funciona con texto en mayúsculas, minúsculas, con o sin acentos
    - Maneja múltiples oraciones y párrafos en el mismo archivo
    - Reconstruye entidades multi-token a partir del esquema BIO
    - Si no encuentra entidades, lo indica explícitamente
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# El modelo por defecto es r=16 — la configuración óptima encontrada
# en la comparación r=8 vs r=16 vs r=32 (ver reporte.pdf, Sección 3)
DEFAULT_LORA_WEIGHTS = "models/ner_lora_r16/lora_weights"
BASE_MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_SEQ_LEN = 64


def _split_into_sentences(text: str) -> list[str]:
    """
    Divide el texto en oraciones de forma robusta.

    Maneja:
        - Múltiples oraciones separadas por punto, salto de línea
        - Párrafos (doble salto de línea)
        - Texto sin puntuación final (se trata como una sola oración)

    No usa un sentence-splitter de NLP completo a propósito — para
    texto financiero corto (comunicados, contratos) un split simple
    por puntuación + saltos de línea es suficiente y no introduce
    dependencias adicionales.
    """
    text = text.replace("\n", ". ")
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip(" .") for s in sentences if s.strip(" .")]
    return sentences if sentences else [text.strip()]


def _tokenize_simple(sentence: str) -> list[str]:
    """
    Tokenización a nivel de palabra (NO subtokens BPE) — coincide con
    el formato del dataset de entrenamiento, donde "tokens" son palabras
    separadas por espacio (eventualmente con puntuación pegada).

    Separa puntuación pegada (comas, paréntesis) pero preserva números
    con decimales, porcentajes y abreviaciones como "91" o "11.25%".
    """
    sentence = re.sub(r"([,;:()\[\]])", r" \1 ", sentence)
    tokens = sentence.split()
    return [t for t in tokens if t]


def load_model(lora_weights_path: str):
    """Carga el modelo BETO+LoRA fine-tuneado desde una ruta local."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    weights_path = Path(lora_weights_path)
    if not weights_path.exists():
        print(
            f"ERROR: no se encontró el modelo en '{lora_weights_path}'. "
            f"Verifica que la carpeta exista e incluya adapter_model.safetensors.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    label_list = [
        "O",
        "B-INSTR", "I-INSTR",
        "B-MONTO", "I-MONTO",
        "B-PLAZO", "I-PLAZO",
        "B-TASA", "I-TASA",
        "B-ENTIDAD", "I-ENTIDAD",
    ]
    id2label = {i: label for i, label in enumerate(label_list)}
    label2id = {label: i for i, label in enumerate(label_list)}

    tokenizer = AutoTokenizer.from_pretrained(str(weights_path))
    base_model = AutoModelForTokenClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(base_model, str(weights_path))
    model.to(device)
    model.eval()

    return model, tokenizer, device


def predict_tags(model, tokenizer, device: str, tokens: list[str]) -> list[str]:
    """
    Predice las etiquetas BIO para una lista de palabras ya tokenizadas.

    Usa word_ids() para mapear de vuelta de subtokens a palabras
    originales — la misma lógica de alineación usada en entrenamiento,
    pero en dirección inversa (subtoken → palabra en lugar de
    palabra → subtoken).
    """
    import torch

    if not tokens:
        return []

    inputs = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pred_ids = outputs.logits[0].argmax(dim=-1).cpu().tolist()
    word_ids = inputs.word_ids()

    id2label = model.config.id2label
    tags: list[str] = ["O"] * len(tokens)
    seen_words: set[int] = set()

    for word_id, pred_id in zip(word_ids, pred_ids):
        if word_id is None or word_id in seen_words:
            continue
        tags[word_id] = id2label[pred_id]
        seen_words.add(word_id)

    return tags


def extract_entities(tokens: list[str], tags: list[str]) -> list[tuple[str, str]]:
    """
    Reconstruye entidades completas a partir del esquema BIO.

    Junta tokens consecutivos B-X seguido de I-X en una sola entidad.
    Maneja correctamente el caso de un I-X que aparece sin un B-X
    precedente (error del modelo) tratándolo como inicio de entidad
    de todos modos — más robusto ante predicciones imperfectas.

    Returns
    -------
    list[tuple[str, str]]
        Lista de (texto_entidad, tipo_entidad).
    """
    entities: list[tuple[str, str]] = []
    current_tokens: list[str] = []
    current_type: str | None = None

    for token, tag in zip(tokens, tags):
        if tag.startswith("B-"):
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = tag[2:]
        elif tag.startswith("I-"):
            entity_type = tag[2:]
            if current_tokens and current_type == entity_type:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append((" ".join(current_tokens), current_type))
                current_tokens = [token]
                current_type = entity_type
        else:  # tag == "O"
            if current_tokens:
                entities.append((" ".join(current_tokens), current_type))
                current_tokens = []
                current_type = None

    if current_tokens:
        entities.append((" ".join(current_tokens), current_type))

    return entities


def run_inference(file_path: str, lora_weights_path: str = DEFAULT_LORA_WEIGHTS) -> None:
    """Función principal: lee el archivo, corre el modelo, imprime entidades."""
    path = Path(file_path)
    if not path.exists():
        print(f"ERROR: el archivo '{file_path}' no existe.", file=sys.stderr)
        sys.exit(1)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="latin-1")

    if not text.strip():
        print("Sin entidades detectadas (archivo vacío).")
        return

    model, tokenizer, device = load_model(lora_weights_path)

    sentences = _split_into_sentences(text)

    all_entities: list[tuple[str, str]] = []
    for sentence in sentences:
        tokens = _tokenize_simple(sentence)
        if not tokens:
            continue
        tags = predict_tags(model, tokenizer, device, tokens)
        entities = extract_entities(tokens, tags)
        all_entities.extend(entities)

    if not all_entities:
        print("Sin entidades detectadas.")
        return

    for entity_text, entity_type in all_entities:
        print(f"{entity_text} {entity_type}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python inference.py ruta/al/archivo.txt", file=sys.stderr)
        sys.exit(1)

    file_path = sys.argv[1]
    run_inference(file_path)


if __name__ == "__main__":
    main()
