"""
scripts/evaluate.py — Evaluación final en test set con seqeval.

Carga los modelos LoRA entrenados (r=8, r=16, r=32) y evalúa cada uno
en el test set (vocabulario heldout, nunca visto en train ni val).

Genera:
    - models/seqeval_scores.json: P/R/F1 global y por entidad para cada r
    - Análisis de errores impreso a consola (qué confunde el modelo)
    - Ejemplos de predicciones en oraciones del test set

Uso::

    python scripts/evaluate.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.evaluate")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch


def load_lora_model_for_inference(lora_weights_path: str, device: str = "cpu"):
    """Carga un modelo BETO+LoRA guardado, listo para inferencia."""
    from peft import PeftModel
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from financial_ner.data.labels import ID_TO_LABEL, LABEL_LIST, LABEL_TO_ID

    tokenizer = AutoTokenizer.from_pretrained(lora_weights_path)
    base_model = AutoModelForTokenClassification.from_pretrained(
        "dccuchile/bert-base-spanish-wwm-cased",
        num_labels=len(LABEL_LIST),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        use_safetensors=True,
    )
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.to(device)
    model.eval()
    return model, tokenizer


def evaluate_on_test(model, tokenizer, test_data: list[dict], device: str = "cpu"):
    """
    Corre el modelo sobre todo el test set y devuelve predicciones
    e ids verdaderos alineados, listos para seqeval.
    """
    from financial_ner.data.alignment import align_dataset

    aligned = align_dataset(test_data, tokenizer, max_length=64)

    all_logits = []
    all_label_ids = []

    with torch.no_grad():
        for example in aligned:
            input_ids = torch.tensor([example["input_ids"]]).to(device)
            attention_mask = torch.tensor([example["attention_mask"]]).to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            all_logits.append(outputs.logits[0].cpu().numpy())
            all_label_ids.append(example["labels"])

    predictions = np.stack(all_logits)
    label_ids = np.array(all_label_ids)
    return predictions, label_ids


def show_prediction_examples(model, tokenizer, test_data: list[dict], n: int = 5, device: str = "cpu"):
    """Imprime ejemplos de predicciones con sus entidades reconstruidas."""
    from financial_ner.data.labels import ID_TO_LABEL

    logger.info("\n── Ejemplos de predicciones en test set:")

    for example in test_data[:n]:
        text = " ".join(example["tokens"])
        inputs = tokenizer(
            example["tokens"], is_split_into_words=True,
            return_tensors="pt", truncation=True, max_length=64,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        pred_ids = outputs.logits[0].argmax(dim=-1).cpu().numpy()

        word_ids = inputs.word_ids()
        pred_labels = []
        prev_word_id = None
        for word_id, pred_id in zip(word_ids, pred_ids):
            if word_id is None or word_id == prev_word_id:
                continue
            pred_labels.append(ID_TO_LABEL[int(pred_id)])
            prev_word_id = word_id

        logger.info("  Texto: %s", text)
        logger.info("  Real:  %s", " ".join(example["ner_tags"]))
        logger.info("  Pred:  %s", " ".join(pred_labels))
        logger.info("")


def main() -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", DEVICE)

    with Path("data/processed/test.json").open(encoding="utf-8") as f:
        test_data = json.load(f)
    logger.info("Test set: %d ejemplos (vocabulario heldout)", len(test_data))

    from financial_ner.evaluation.metrics import (
        full_classification_report,
        per_entity_scores,
        predictions_to_labels,
    )
    from seqeval.metrics import f1_score, precision_score, recall_score

    all_results = {}

    for r in [8, 16, 32]:
        lora_path = f"models/ner_lora_r{r}/lora_weights"
        if not Path(lora_path).exists():
            logger.warning("No encontrado: %s — saltando r=%d", lora_path, r)
            continue

        logger.info("\n%s", "=" * 65)
        logger.info("  EVALUACIÓN EN TEST SET — LoRA r=%d", r)
        logger.info("%s", "=" * 65)

        model, tokenizer = load_lora_model_for_inference(lora_path, DEVICE)
        predictions, label_ids = evaluate_on_test(model, tokenizer, test_data, DEVICE)

        true_labels, pred_labels = predictions_to_labels(predictions, label_ids)
        global_f1 = f1_score(true_labels, pred_labels)
        global_precision = precision_score(true_labels, pred_labels)
        global_recall = recall_score(true_labels, pred_labels)

        logger.info(
            "Global — Precision: %.4f | Recall: %.4f | F1: %.4f",
            global_precision, global_recall, global_f1,
        )

        per_entity = per_entity_scores(predictions, label_ids)
        logger.info("\nPor tipo de entidad:")
        for entity, scores in per_entity.items():
            logger.info(
                "  %-10s P=%.3f R=%.3f F1=%.3f",
                entity, scores["precision"], scores["recall"], scores["f1"],
            )

        report = full_classification_report(predictions, label_ids)
        logger.info("\nReporte completo (seqeval):\n%s", report)

        if r == 16:  # mostrar ejemplos solo para el modelo óptimo
            show_prediction_examples(model, tokenizer, test_data, n=5, device=DEVICE)

        all_results[f"r{r}"] = {
            "test_precision": global_precision,
            "test_recall": global_recall,
            "test_f1": global_f1,
            "per_entity": per_entity,
        }

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    output_path = Path("models/seqeval_scores.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("\n✓ Resultados guardados: %s", output_path)

    logger.info("\n%s", "=" * 65)
    logger.info("  RESUMEN FINAL — TEST SET (vocabulario heldout)")
    logger.info("%s", "=" * 65)
    for r_key, results in all_results.items():
        logger.info(
            "  %-5s F1=%.4f  P=%.4f  R=%.4f",
            r_key, results["test_f1"], results["test_precision"], results["test_recall"],
        )
    logger.info("%s", "=" * 65)


if __name__ == "__main__":
    main()
