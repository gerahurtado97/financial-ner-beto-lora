"""
tests/unit/test_generate_dataset.py — Tests del generador del dataset completo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.generate_dataset import deduplicate, generate_dataset
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
