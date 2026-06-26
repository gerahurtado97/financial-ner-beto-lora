"""
scripts/generate_dataset.py — Genera train/val/test con vocabulario disjunto.

CORRECCIÓN CRÍTICA respecto a la versión anterior:
    El dataset ya NO se genera como un solo archivo para luego dividir
    aleatoriamente. Se generan train.json (vocab_pool="train") y
    val.json/test.json (vocab_pool="heldout") por separado, usando
    palabras DISTINTAS en cada uno. Esto evita que el modelo memorice
    vocabulario en lugar de aprender el patrón BIO — el problema que
    causó eval_f1=1.0 en el primer experimento (ver reporte).

También guarda data/raw/financial_ner_dataset.json con TODO el contenido
combinado (train+val+test) únicamente como referencia/documentación del
dataset completo — NO se usa para re-derivar los splits.

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

from financial_ner.data.generate_dataset import generate_train_val_test
from financial_ner.data.schema import label_distribution, validate_dataset


def main() -> None:
    raw_path = Path("data/raw/financial_ner_dataset.json")
    processed_path = Path("data/processed")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generando train/val/test con vocabulario DISJUNTO...")
    logger.info("  train -> vocab_pool='train'")
    logger.info("  val, test -> vocab_pool='heldout' (palabras nunca vistas en train)")

    train, val, test = generate_train_val_test(
        n_per_template_train=35,
        n_negative_train=50,
        n_per_template_evalsplit=8,
        n_negative_evalsplit=12,
        seed=42,
    )

    for name, split in [("train", train), ("val", val), ("test", test)]:
        validate_dataset(split)
        logger.info("OK %s validado: %d ejemplos", name, len(split))

    from financial_ner.data.vocabularies import INSTRUMENTOS_HELDOUT, INSTRUMENTOS_TRAIN

    train_vocab = set(INSTRUMENTOS_TRAIN)
    heldout_vocab = set(INSTRUMENTOS_HELDOUT)
    assert train_vocab.isdisjoint(heldout_vocab), "Vocabularios INSTR no son disjuntos"
    logger.info("OK Vocabulario INSTR train/heldout confirmado disjunto")

    for name, split in [("train", train), ("val", val), ("test", test)]:
        out_path = processed_path / f"{name}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(split, f, ensure_ascii=False, indent=2)
        logger.info("OK %s guardado: %s", name, out_path)

    combined = train + val + test
    with raw_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    logger.info("OK Dataset combinado (referencia): %s (%d ejemplos)", raw_path, len(combined))

    logger.info("\nDistribucion de entidades por split:")
    for name, split in [("train", train), ("val", val), ("test", test)]:
        dist = label_distribution(split)
        entity_counts = {k: v for k, v in dist.items() if k.startswith("B-") and v > 0}
        logger.info("  %-5s: %s", name, entity_counts)


if __name__ == "__main__":
    main()
