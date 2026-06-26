"""
tests/unit/test_schema.py — Tests del validador de esquema BIO.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.schema import (
    SchemaValidationError,
    label_distribution,
    validate_dataset,
    validate_example,
)


class TestValidateExample:
    def test_valid_example_passes(self):
        example = {"tokens": ["Los", "CETES"], "ner_tags": ["O", "B-INSTR"]}
        validate_example(example, 0)  # no debe lanzar

    def test_missing_tokens_key_raises(self):
        example = {"ner_tags": ["O"]}
        with pytest.raises(SchemaValidationError, match="tokens.*ner_tags"):
            validate_example(example, 0)

    def test_length_mismatch_raises(self):
        example = {"tokens": ["a", "b", "c"], "ner_tags": ["O", "O"]}
        with pytest.raises(SchemaValidationError, match="longitudes distintas"):
            validate_example(example, 0)

    def test_empty_tokens_raises(self):
        example = {"tokens": [], "ner_tags": []}
        with pytest.raises(SchemaValidationError, match="vacío"):
            validate_example(example, 0)

    def test_invalid_label_raises(self):
        example = {"tokens": ["a"], "ner_tags": ["B-FAKE"]}
        with pytest.raises(SchemaValidationError, match="inválidas"):
            validate_example(example, 0)

    def test_i_tag_without_b_raises(self):
        """I-INSTR sin B-INSTR precedente es BIO inválido."""
        example = {"tokens": ["a", "b"], "ner_tags": ["O", "I-INSTR"]}
        with pytest.raises(SchemaValidationError, match="precedente"):
            validate_example(example, 0)

    def test_i_tag_with_different_entity_raises(self):
        """B-INSTR seguido de I-PLAZO (entidad distinta) es inválido."""
        example = {"tokens": ["a", "b"], "ner_tags": ["B-INSTR", "I-PLAZO"]}
        with pytest.raises(SchemaValidationError, match="precedente"):
            validate_example(example, 0)

    def test_valid_multi_token_entity(self):
        """B-PLAZO + I-PLAZO es válido (entidad de 2 tokens)."""
        example = {"tokens": ["91", "días"], "ner_tags": ["B-PLAZO", "I-PLAZO"]}
        validate_example(example, 0)  # no debe lanzar

    def test_all_o_is_valid(self):
        example = {"tokens": ["el", "clima"], "ner_tags": ["O", "O"]}
        validate_example(example, 0)


class TestValidateDataset:
    def test_valid_dataset_passes(self):
        dataset = [
            {"tokens": ["Los", "CETES"], "ner_tags": ["O", "B-INSTR"]},
            {"tokens": ["Banxico"], "ner_tags": ["B-ENTIDAD"]},
        ]
        validate_dataset(dataset)

    def test_empty_dataset_raises(self):
        with pytest.raises(SchemaValidationError, match="vacío"):
            validate_dataset([])

    def test_one_bad_example_fails_whole_dataset(self):
        dataset = [
            {"tokens": ["a"], "ner_tags": ["O"]},
            {"tokens": ["b", "c"], "ner_tags": ["O"]},  # longitud distinta
        ]
        with pytest.raises(SchemaValidationError):
            validate_dataset(dataset)


class TestLabelDistribution:
    def test_counts_all_labels(self):
        dataset = [
            {"tokens": ["a", "b"], "ner_tags": ["O", "B-INSTR"]},
            {"tokens": ["c"], "ner_tags": ["B-INSTR"]},
        ]
        dist = label_distribution(dataset)
        assert dist["B-INSTR"] == 2
        assert dist["O"] == 1

    def test_zero_for_unused_labels(self):
        dataset = [{"tokens": ["a"], "ner_tags": ["O"]}]
        dist = label_distribution(dataset)
        assert dist["B-MONTO"] == 0
