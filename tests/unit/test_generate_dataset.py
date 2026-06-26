"""
tests/unit/test_generate_dataset.py — Tests del generador del dataset completo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.generate_dataset import (
    deduplicate,
    generate_dataset,
    generate_train_val_test,
)
from financial_ner.data.schema import validate_dataset


class TestGenerateDataset:
    def test_produces_valid_bio_dataset(self):
        dataset = generate_dataset(n_per_template=5, n_negative=5, seed=0)
        validate_dataset(dataset)  # no debe lanzar

    def test_reproducible_with_same_seed(self):
        ds1 = generate_dataset(n_per_template=5, n_negative=5, seed=42)
        ds2 = generate_dataset(n_per_template=5, n_negative=5, seed=42)
        assert ds1 == ds2

    def test_different_seeds_differ(self):
        ds1 = generate_dataset(n_per_template=5, n_negative=5, seed=1)
        ds2 = generate_dataset(n_per_template=5, n_negative=5, seed=2)
        assert ds1 != ds2

    def test_includes_negative_examples(self):
        dataset = generate_dataset(n_per_template=5, n_negative=10, seed=0)
        all_o = [ex for ex in dataset if all(t == "O" for t in ex["ner_tags"])]
        assert len(all_o) == 10

    def test_n_per_template_scales_dataset_size(self):
        small = generate_dataset(n_per_template=5, n_negative=0, seed=0)
        large = generate_dataset(n_per_template=10, n_negative=0, seed=0)
        assert len(large) > len(small)

    def test_all_examples_have_required_keys(self):
        dataset = generate_dataset(n_per_template=3, n_negative=2, seed=0)
        for ex in dataset:
            assert "tokens" in ex
            assert "ner_tags" in ex


class TestDeduplicate:
    def test_removes_exact_duplicates(self):
        dataset = [
            {"tokens": ["a", "b"], "ner_tags": ["O", "O"]},
            {"tokens": ["a", "b"], "ner_tags": ["O", "O"]},  # duplicado exacto
            {"tokens": ["c"], "ner_tags": ["O"]},
        ]
        result = deduplicate(dataset)
        assert len(result) == 2

    def test_keeps_unique_examples(self):
        dataset = [
            {"tokens": ["a"], "ner_tags": ["O"]},
            {"tokens": ["b"], "ner_tags": ["B-INSTR"]},
        ]
        result = deduplicate(dataset)
        assert len(result) == 2

    def test_empty_dataset(self):
        assert deduplicate([]) == []


def _extract_spans(dataset: list[dict], entity_type: str) -> set[str]:
    """Extrae el conjunto de spans de texto de un tipo de entidad."""
    spans: set[str] = set()
    for ex in dataset:
        tokens, tags = ex["tokens"], ex["ner_tags"]
        current: list[str] = []
        for tok, tag in zip(tokens, tags):
            if tag == f"B-{entity_type}":
                if current:
                    spans.add(" ".join(current))
                current = [tok]
            elif tag == f"I-{entity_type}" and current:
                current.append(tok)
            else:
                if current:
                    spans.add(" ".join(current))
                    current = []
        if current:
            spans.add(" ".join(current))
    return spans


class TestVocabPool:
    """
    Tests de la corrección anti-memorización: vocab_pool="train" vs "heldout"
    deben producir vocabulario DISJUNTO. Esta es la corrección crítica que
    evita que el modelo memorice palabras en lugar de aprender el patrón BIO.
    """

    def test_train_pool_differs_from_heldout_pool(self):
        train_ds = generate_dataset(n_per_template=10, n_negative=5, seed=0, vocab_pool="train")
        heldout_ds = generate_dataset(
            n_per_template=10, n_negative=5, seed=0, vocab_pool="heldout"
        )

        for entity in ["INSTR", "ENTIDAD", "PLAZO", "TASA", "MONTO"]:
            train_spans = _extract_spans(train_ds, entity)
            heldout_spans = _extract_spans(heldout_ds, entity)
            overlap = train_spans & heldout_spans
            assert len(overlap) == 0, (
                f"Vocabulario de {entity} NO es disjunto entre train y heldout: {overlap}"
            )

    def test_invalid_vocab_pool_raises(self):
        with pytest.raises(ValueError, match="vocab_pool"):
            generate_dataset(n_per_template=1, n_negative=1, vocab_pool="invalid")  # type: ignore[arg-type]

    def test_default_vocab_pool_is_train(self):
        """Por compatibilidad, sin especificar vocab_pool debe usar 'train'."""
        ds_default = generate_dataset(n_per_template=5, n_negative=2, seed=0)
        ds_explicit = generate_dataset(n_per_template=5, n_negative=2, seed=0, vocab_pool="train")
        assert ds_default == ds_explicit


class TestGenerateTrainValTest:
    def test_returns_three_splits(self):
        train, val, test = generate_train_val_test(
            n_per_template_train=5, n_negative_train=3,
            n_per_template_evalsplit=3, n_negative_evalsplit=2,
            seed=0,
        )
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_all_splits_pass_bio_validation(self):
        train, val, test = generate_train_val_test(
            n_per_template_train=5, n_negative_train=3,
            n_per_template_evalsplit=3, n_negative_evalsplit=2,
            seed=0,
        )
        validate_dataset(train)
        validate_dataset(val)
        validate_dataset(test)

    def test_train_and_heldout_vocab_disjoint_end_to_end(self):
        """
        Test end-to-end de la corrección anti-overfitting: el vocabulario
        de entidades en train NUNCA debe aparecer en val o test.
        """
        train, val, test = generate_train_val_test(
            n_per_template_train=20, n_negative_train=10,
            n_per_template_evalsplit=10, n_negative_evalsplit=5,
            seed=42,
        )

        for entity in ["INSTR", "ENTIDAD", "PLAZO", "TASA", "MONTO"]:
            train_spans = _extract_spans(train, entity)
            val_spans = _extract_spans(val, entity)
            test_spans = _extract_spans(test, entity)

            assert train_spans.isdisjoint(val_spans), (
                f"{entity}: vocabulario compartido entre train y val: "
                f"{train_spans & val_spans}"
            )
            assert train_spans.isdisjoint(test_spans), (
                f"{entity}: vocabulario compartido entre train y test: "
                f"{train_spans & test_spans}"
            )

    def test_val_and_test_are_deduplicated_internally(self):
        train, val, test = generate_train_val_test(
            n_per_template_train=5, n_negative_train=3,
            n_per_template_evalsplit=5, n_negative_evalsplit=3,
            seed=0,
        )
        val_keys = [tuple(ex["tokens"]) for ex in val]
        assert len(val_keys) == len(set(val_keys)), "val contiene duplicados internos"
