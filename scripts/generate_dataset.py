"""
scripts/generate_dataset.py — Genera el dataset sintético y lo guarda en JSON.

Produce data/raw/financial_ner_dataset.json con el formato especificado
en el enunciado del proyecto:

    {"tokens": ["Los","CETES","a","91","dias"],
     "ner_tags": ["O","B-INSTR","O","B-PLAZO","I-PLAZO"]}

Uso::

    python scripts/generate_dataset.py
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
logger = logging.getLogger("scripts.generate_dataset")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_ner.data.generate_dataset import deduplicate, generate_dataset
from financial_ner.data.schema import label_distribution, validate_dataset


def main() -> None:
    output_path = Path("data/raw/financial_ner_dataset.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generando dataset sintético de NER financiero...")
    dataset = generate_dataset(n_per_template=35, n_negative=50, seed=42)
    dataset = deduplicate(dataset)

    logger.info("Validando esquema BIO...")
    validate_dataset(dataset)

    dist = label_distribution(dataset)
    logger.info("Distribución de etiquetas:")
    for label, count in sorted(dist.items(), key=lambda x: -x[1]):
        if count > 0:
            logger.info("  %-12s %d", label, count)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    size_kb = output_path.stat().st_size / 1024
    logger.info(
        "✓ Dataset guardado: %s (%d ejemplos, %.1f KB)",
        output_path, len(dataset), size_kb,
    )


if __name__ == "__main__":
    main()
