"""
scripts/prepare_data.py — Etapa de validacion del pipeline DVC.

NOTA: el split train/val/test con vocabulario disjunto ya se realiza
en scripts/generate_dataset.py (que genera train.json, val.json y
test.json directamente en data/processed/). Este script simplemente
valida que esos archivos existen y son correctos en esquema BIO,
sirviendo como stage explicito del pipeline DVC.

Uso::

    python scripts/prepare_data.py
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
logger = logging.getLogger("scripts.prepare_data")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_ner.data.schema import label_distribution, validate_dataset


def main() -> None:
    processed_path = Path("data/processed")

    for name in ["train", "val", "test"]:
        path = processed_path / f"{name}.json"
        if not path.exists():
            logger.error(
                "%s no encontrado. Ejecuta primero: python scripts/generate_dataset.py",
                path,
            )
            sys.exit(1)

        with path.open(encoding="utf-8") as f:
            split = json.load(f)

        validate_dataset(split)
        dist = label_distribution(split)
        entity_counts = {k: v for k, v in dist.items() if k.startswith("B-") and v > 0}
        logger.info("OK %-5s: %d ejemplos | entidades: %s", name, len(split), entity_counts)

    logger.info("Pipeline de datos validado correctamente.")


if __name__ == "__main__":
    main()
