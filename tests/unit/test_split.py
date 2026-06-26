"""
tests/unit/test_split.py — Tests del split train/val/test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.split import train_val_test_split


def _make_dataset(n: int) -> list[dict]:
    return [{"tokens": [f"tok{i}"], "ner_tags": ["O"]} for i in range(n)]


class TestTrainValTestSplit:
    def test_proportions_approximately_correct(self):
        dataset = _make_dataset(100)
        train, val, test = train_val_test_split(dataset, 0.70, 0.15, 0.15, seed=42)
        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15

    def test_no_overlap_between_splits(self):
        dataset = _make_dataset(50)
        train, val, test = train_val_test_split(dataset, 0.70, 0.15, 0.15, seed=1)
        train_toks = {tuple(ex["tokens"]) for ex in train}
        val_toks = {tuple(ex["tokens"]) for ex in val}
        test_toks = {tuple(ex["tokens"]) for ex in test}
        assert train_toks.isdisjoint(val_toks)
        assert train_toks.isdisjoint(test_toks)
        assert val_toks.isdisjoint(test_toks)

    def test_all_examples_preserved(self):
        dataset = _make_dataset(50)
        train, val, test = train_val_test_split(dataset, 0.70, 0.15, 0.15, seed=1)
        assert len(train) + len(val) + len(test) == len(dataset)

    def test_reproducible_with_same_seed(self):
        dataset = _make_dataset(30)
        t1, v1, te1 = train_val_test_split(dataset, seed=7)
        t2, v2, te2 = train_val_test_split(dataset, seed=7)
        assert t1 == t2
        assert v1 == v2
        assert te1 == te2

    def test_invalid_proportions_raise(self):
        dataset = _make_dataset(10)
        with pytest.raises(ValueError, match="sumar 1.0"):
            train_val_test_split(dataset, 0.5, 0.3, 0.3)
