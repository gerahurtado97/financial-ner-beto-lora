"""
tests/unit/test_alignment.py — Tests de la alineación de etiquetas BIO con subtokens.

Esta es la pieza más crítica del proyecto técnicamente (señalada en el
enunciado). Usamos un tokenizer real de BETO para verificar el
comportamiento exacto con WordPiece — no un mock, porque el detalle
de cómo word_ids() se comporta es justamente lo que queremos validar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.alignment import (
    align_dataset,
    align_labels_with_tokens,
    count_subtoken_splits,
)
from financial_ner.data.labels import IGNORE_INDEX, LABEL_TO_ID


@pytest.fixture(scope="module")
def beto_tokenizer():
    """Carga el tokenizador real de BETO — se reusa en todo el módulo."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")


class TestAlignLabelsWithTokens:
    def test_output_has_same_length_as_input_ids(self, beto_tokenizer):
        """labels debe tener exactamente la misma longitud que input_ids."""
        tokens = ["Los", "CETES", "a", "91", "días"]
        tags = ["O", "B-INSTR", "O", "B-PLAZO", "I-PLAZO"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=32)
        assert len(encoding["labels"]) == len(encoding["input_ids"])

    def test_special_tokens_get_ignore_index(self, beto_tokenizer):
        """[CLS] y [SEP] deben recibir IGNORE_INDEX, no una etiqueta real."""
        tokens = ["Banxico"]
        tags = ["B-ENTIDAD"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=16)
        # El primer label corresponde a [CLS] — debe ser IGNORE_INDEX
        assert encoding["labels"][0] == IGNORE_INDEX

    def test_padding_gets_ignore_index(self, beto_tokenizer):
        """Las posiciones de padding deben recibir IGNORE_INDEX."""
        tokens = ["Banxico"]
        tags = ["B-ENTIDAD"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=32)
        # Las últimas posiciones son padding — deben ser todas IGNORE_INDEX
        assert encoding["labels"][-1] == IGNORE_INDEX

    def test_single_subtoken_word_gets_its_label(self, beto_tokenizer):
        """
        Una palabra que NO se divide (1 subtoken) debe recibir su
        etiqueta real, sin ningún IGNORE_INDEX intermedio para ella.
        """
        tokens = ["Banxico", "mantuvo"]
        tags = ["B-ENTIDAD", "O"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=16)
        non_ignore = [
            lbl for lbl in encoding["labels"] if lbl != IGNORE_INDEX
        ]
        # Ambas palabras deben tener exactamente una etiqueta real
        assert len(non_ignore) == 2
        assert LABEL_TO_ID["B-ENTIDAD"] in non_ignore
        assert LABEL_TO_ID["O"] in non_ignore

    def test_multi_subtoken_word_only_first_gets_label(self, beto_tokenizer):
        """
        CASO CRÍTICO: cuando una palabra se divide en N subtokens,
        SOLO el primero recibe la etiqueta — los demás IGNORE_INDEX.

        Verificamos esto encontrando una palabra que BETO efectivamente
        divide en más de un subtoken.
        """
        # Buscar una palabra del vocabulario financiero que se divida
        word = "UDIBONOS"
        n_subtokens = len(beto_tokenizer.tokenize(word))

        tokens = ["Los", word, "rinden"]
        tags = ["O", "B-INSTR", "O"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=16)

        # Contar cuántas veces aparece B-INSTR en las etiquetas alineadas
        # Debe aparecer EXACTAMENTE 1 vez, sin importar en cuántos
        # subtokens se dividió la palabra
        n_b_instr = sum(
            1 for lbl in encoding["labels"] if lbl == LABEL_TO_ID["B-INSTR"]
        )
        assert n_b_instr == 1, (
            f"'{word}' se dividió en {n_subtokens} subtokens, "
            f"pero B-INSTR aparece {n_b_instr} veces (debe ser exactamente 1)"
        )

    def test_multi_token_entity_alignment(self, beto_tokenizer):
        """
        Una entidad de múltiples PALABRAS (ej. '91 días' = B-PLAZO + I-PLAZO)
        debe preservar ambas etiquetas tras la alineación a subtokens,
        independientemente de cuántos subtokens use cada palabra.
        """
        tokens = ["el", "plazo", "es", "91", "días"]
        tags = ["O", "O", "O", "B-PLAZO", "I-PLAZO"]
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=16)

        n_b_plazo = sum(1 for lbl in encoding["labels"] if lbl == LABEL_TO_ID["B-PLAZO"])
        n_i_plazo = sum(1 for lbl in encoding["labels"] if lbl == LABEL_TO_ID["I-PLAZO"])

        assert n_b_plazo == 1
        assert n_i_plazo == 1

    def test_raises_on_length_mismatch(self, beto_tokenizer):
        with pytest.raises(ValueError, match="misma longitud"):
            align_labels_with_tokens(["a", "b"], ["O"], beto_tokenizer)

    def test_truncation_respects_max_length(self, beto_tokenizer):
        """Con max_length pequeño, la salida no debe exceder ese límite."""
        tokens = ["Banxico"] * 50  # secuencia artificialmente larga
        tags = ["B-ENTIDAD"] * 50
        encoding = align_labels_with_tokens(tokens, tags, beto_tokenizer, max_length=16)
        assert len(encoding["input_ids"]) == 16
        assert len(encoding["labels"]) == 16


class TestAlignDataset:
    def test_aligns_full_dataset(self, beto_tokenizer):
        dataset = [
            {"tokens": ["Banxico", "elevó"], "ner_tags": ["B-ENTIDAD", "O"]},
            {"tokens": ["Los", "CETES"], "ner_tags": ["O", "B-INSTR"]},
        ]
        aligned = align_dataset(dataset, beto_tokenizer, max_length=16)
        assert len(aligned) == 2
        for example in aligned:
            assert "input_ids" in example
            assert "attention_mask" in example
            assert "labels" in example
            assert len(example["input_ids"]) == len(example["labels"])

    def test_empty_dataset(self, beto_tokenizer):
        assert align_dataset([], beto_tokenizer) == []


class TestCountSubtokenSplits:
    def test_counts_words_correctly(self, beto_tokenizer):
        dataset = [{"tokens": ["Banxico", "mantuvo"], "ner_tags": ["B-ENTIDAD", "O"]}]
        stats = count_subtoken_splits(dataset, beto_tokenizer)
        assert stats["total_words"] == 2
        assert "split_words" in stats
        assert "split_ratio" in stats
        assert 0.0 <= stats["split_ratio"] <= 1.0

    def test_ratio_is_consistent(self, beto_tokenizer):
        dataset = [{"tokens": ["a", "b", "c", "d"], "ner_tags": ["O"] * 4}]
        stats = count_subtoken_splits(dataset, beto_tokenizer)
        expected_ratio = round(stats["split_words"] / stats["total_words"], 4)
        assert stats["split_ratio"] == expected_ratio
