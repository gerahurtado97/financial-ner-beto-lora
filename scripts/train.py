"""
scripts/train.py — Fine-tuning de BETO + LoRA para NER financiero.

Uso::

    python scripts/train.py --lora-r 8 --lora-alpha 16
    python scripts/train.py --lora-r 16 --lora-alpha 32
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scripts.train")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tuning BETO + LoRA para NER financiero")
    p.add_argument("--config", default="configs/model_config.yaml")
    p.add_argument("--lora-r", type=int, required=True, help="Rango LoRA (8 o 16)")
    p.add_argument("--lora-alpha", type=int, required=True, help="Alpha LoRA")
    return p.parse_args()


def main() -> None:
    import yaml
    from datasets import Dataset
    from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 65)
    logger.info("  NER Financiero — BETO + LoRA (r=%d)", args.lora_r)
    logger.info("=" * 65)
    logger.info("  Device: %s", DEVICE)
    if DEVICE.type == "cuda":
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    # ── 1. Cargar datos procesados
    from financial_ner.data.alignment import align_dataset, count_subtoken_splits
    from financial_ner.model.lora_model import build_lora_model, count_trainable_parameters

    data_cfg = config["data"]
    processed_path = Path(data_cfg["processed_path"])

    with (processed_path / "train.json").open(encoding="utf-8") as f:
        train_raw = json.load(f)
    with (processed_path / "val.json").open(encoding="utf-8") as f:
        val_raw = json.load(f)

    logger.info("Datos: train=%d val=%d", len(train_raw), len(val_raw))

    # ── 2. Construir modelo LoRA
    model_cfg = config["model"]
    model, tokenizer = build_lora_model(
        base_model_name=model_cfg["base_model"],
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=config["lora"]["lora_dropout"],
        target_modules=config["lora"]["target_modules"],
    )
    model.to(DEVICE)

    param_counts = count_trainable_parameters(model)
    logger.info(
        "Parámetros: %s entrenables / %s totales (%.3f%%)",
        f"{param_counts['trainable']:,}", f"{param_counts['total']:,}",
        param_counts["trainable_pct"],
    )

    # Estadística de subtokens — para el reporte
    split_stats = count_subtoken_splits(train_raw, tokenizer)
    logger.info(
        "Subtokenización: %d/%d palabras (%.1f%%) se dividen en múltiples subtokens",
        split_stats["split_words"], split_stats["total_words"],
        split_stats["split_ratio"] * 100,
    )

    # ── 3. Alinear etiquetas a subtokens
    max_len = data_cfg["max_seq_len"]
    train_aligned = align_dataset(train_raw, tokenizer, max_length=max_len)
    val_aligned = align_dataset(val_raw, tokenizer, max_length=max_len)

    train_dataset = Dataset.from_list(train_aligned)
    val_dataset = Dataset.from_list(val_aligned)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ── 4. Entrenamiento
    from financial_ner.evaluation.metrics import compute_metrics

    train_cfg = config["training"]
    models_path = Path("models") / f"ner_lora_r{args.lora_r}"
    models_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(models_path / "checkpoints"),
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"] * 2,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_ratio=train_cfg["warmup_ratio"],
        eval_strategy=train_cfg["eval_strategy"],
        save_strategy=train_cfg["save_strategy"],
        metric_for_best_model=train_cfg["metric_for_best_model"],
        load_best_model_at_end=train_cfg["load_best_model_at_end"],
        logging_steps=10,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("\n%s", "─" * 65)
    logger.info("  Iniciando entrenamiento (r=%d, alpha=%d)...", args.lora_r, args.lora_alpha)
    logger.info("  Epochs: %d | Batch: %d | LR: %.4f",
                train_cfg["epochs"], train_cfg["batch_size"], train_cfg["learning_rate"])
    logger.info("%s\n", "─" * 65)

    trainer.train()

    # ── 5. Evaluación final en val
    eval_metrics = trainer.evaluate()
    logger.info("\nMétricas finales en validación:")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            logger.info("  %s: %.4f", k, v)

    # ── 6. Guardar modelo LoRA (solo el adaptador — ligero)
    final_model_path = models_path / "lora_weights"
    model.save_pretrained(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))

    size_mb = sum(
        f.stat().st_size for f in final_model_path.rglob("*") if f.is_file()
    ) / (1024 ** 2)
    logger.info("✓ Pesos LoRA guardados: %s (%.1f MB)", final_model_path, size_mb)

    # ── 7. Guardar métricas para DVC
    metrics_file = Path("models") / f"metrics_r{args.lora_r}.json"
    with metrics_file.open("w") as f:
        json.dump({
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "trainable_params": param_counts["trainable"],
            "total_params": param_counts["total"],
            "trainable_pct": param_counts["trainable_pct"],
            "subtoken_split_ratio": split_stats["split_ratio"],
            "eval_precision": eval_metrics.get("eval_precision", 0),
            "eval_recall": eval_metrics.get("eval_recall", 0),
            "eval_f1": eval_metrics.get("eval_f1", 0),
        }, f, indent=2)

    logger.info("\n%s", "=" * 65)
    logger.info("  RESUMEN — LoRA r=%d", args.lora_r)
    logger.info("=" * 65)
    logger.info("  F1 (val):        %.4f", eval_metrics.get("eval_f1", 0))
    logger.info("  Precision (val): %.4f", eval_metrics.get("eval_precision", 0))
    logger.info("  Recall (val):    %.4f", eval_metrics.get("eval_recall", 0))
    logger.info("  Parámetros entrenables: %.3f%%", param_counts["trainable_pct"])
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
