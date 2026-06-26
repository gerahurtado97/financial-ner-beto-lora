"""
tests/unit/test_labels.py — Tests del mapeo de etiquetas BIO ↔ índices.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.labels import (
    ID_TO_LABEL,
    IGNORE_INDEX,
    LABEL_LIST,
    LABEL_TO_ID,
    ids_to_labels,
    labels_to_ids,
)


class TestLabelMapping:
    def test_label_list_has_11_classes(self):
        """5 entidades × 2 (B/I) + O = 11 clases."""
        assert len(LABEL_LIST) == 11

    def test_o_is_index_zero(self):
        assert LABEL_TO_ID["O"] == 0

    def test_mapping_is_bijective(self):
        for label, idx in LABEL_TO_ID.items():
            assert ID_TO_LABEL[idx] == label

    def test_all_entity_types_present(self):
        for entity in ["INSTR", "MONTO", "PLAZO", "TASA", "ENTIDAD"]:
            assert f"B-{entity}" in LABEL_TO_ID
            assert f"I-{entity}" in LABEL_TO_ID

    def test_ignore_index_is_minus_100(self):
        """Convención estándar de PyTorch CrossEntropyLoss / Hugging Face."""
        assert IGNORE_INDEX == -100


class TestLabelsToIds:
    def test_roundtrip(self):
        labels = ["O", "B-INSTR", "I-INSTR"]
        ids = labels_to_ids(labels)
        recovered = ids_to_labels(ids)
        assert recovered == labels

    def test_ignore_index_maps_to_none(self):
        result = ids_to_labels([IGNORE_INDEX, 0, 1])
        assert result[0] is None
        assert result[1] == "O"
        assert result[2] == "B-INSTR"
