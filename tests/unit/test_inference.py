"""
tests/unit/test_inference.py — Tests de la lógica pura de inference.py.

Probamos extract_entities, _tokenize_simple y _split_into_sentences
sin cargar el modelo (que requiere red/GPU) — son funciones puras
de texto que se pueden verificar de forma aislada y rápida.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inference import _split_into_sentences, _tokenize_simple, extract_entities


class TestExtractEntities:
    def test_enunciado_example_first_sentence(self):
        """
        Caso exacto del enunciado del proyecto (Sección 7):
        'Los CETES a 91 días rinden 11.25 por ciento'
        debe producir: CETES INSTR, 91 días PLAZO, 11.25 por ciento TASA
        """
        tokens = ["Los", "CETES", "a", "91", "días", "rinden", "11.25", "por", "ciento"]
        tags = ["O", "B-INSTR", "O", "B-PLAZO", "I-PLAZO", "O", "B-TASA", "I-TASA", "I-TASA"]
        entities = extract_entities(tokens, tags)
        assert entities == [
            ("CETES", "INSTR"),
            ("91 días", "PLAZO"),
            ("11.25 por ciento", "TASA"),
        ]

    def test_enunciado_example_second_sentence(self):
        """
        'Banxico mantuvo la TIIE sin cambio' → Banxico ENTIDAD, TIIE INSTR
        """
        tokens = ["Banxico", "mantuvo", "la", "TIIE", "sin", "cambio"]
        tags = ["B-ENTIDAD", "O", "O", "B-INSTR", "O", "O"]
        entities = extract_entities(tokens, tags)
        assert entities == [("Banxico", "ENTIDAD"), ("TIIE", "INSTR")]

    def test_no_entities_returns_empty_list(self):
        tokens = ["El", "clima", "estuvo", "soleado"]
        tags = ["O", "O", "O", "O"]
        assert extract_entities(tokens, tags) == []

    def test_single_token_entity(self):
        tokens = ["Banxico"]
        tags = ["B-ENTIDAD"]
        assert extract_entities(tokens, tags) == [("Banxico", "ENTIDAD")]

    def test_multi_token_entity_joins_correctly(self):
        tokens = ["100", "millones", "de", "pesos"]
        tags = ["B-MONTO", "I-MONTO", "I-MONTO", "I-MONTO"]
        assert extract_entities(tokens, tags) == [("100 millones de pesos", "MONTO")]

    def test_i_tag_without_preceding_b_tag_is_robust(self):
        """
        Robustez ante predicciones imperfectas del modelo: un I-X sin
        B-X precedente no debe romper el script, se trata como inicio
        de entidad.
        """
        tokens = ["x", "días"]
        tags = ["O", "I-PLAZO"]
        entities = extract_entities(tokens, tags)
        assert entities == [("días", "PLAZO")]

    def test_entity_ends_at_end_of_sequence(self):
        """Una entidad que llega hasta el último token debe cerrarse correctamente."""
        tokens = ["la", "tasa", "es", "Banxico"]
        tags = ["O", "O", "O", "B-ENTIDAD"]
        entities = extract_entities(tokens, tags)
        assert entities == [("Banxico", "ENTIDAD")]

    def test_two_consecutive_entities_different_types(self):
        """Dos entidades de tipos distintos consecutivas (sin O entre ellas)."""
        tokens = ["CETES", "Banxico"]
        tags = ["B-INSTR", "B-ENTIDAD"]
        entities = extract_entities(tokens, tags)
        assert entities == [("CETES", "INSTR"), ("Banxico", "ENTIDAD")]

    def test_i_tag_different_type_than_current_b_starts_new_entity(self):
        """
        B-INSTR seguido de I-PLAZO (tipo distinto) — predicción
        inconsistente del modelo, debe cerrarse la entidad anterior
        y empezar una nueva en lugar de fusionarlas incorrectamente.
        """
        tokens = ["CETES", "días"]
        tags = ["B-INSTR", "I-PLAZO"]
        entities = extract_entities(tokens, tags)
        assert entities == [("CETES", "INSTR"), ("días", "PLAZO")]


class TestSplitIntoSentences:
    def test_splits_on_period(self):
        text = "Los CETES rinden 11.25%. Banxico mantuvo la TIIE."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2

    def test_single_sentence_no_punctuation(self):
        text = "Los CETES rinden 11.25 por ciento"
        sentences = _split_into_sentences(text)
        assert len(sentences) == 1

    def test_handles_newlines(self):
        text = "Los CETES rinden 11.25%\nBanxico mantuvo la TIIE"
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2

    def test_empty_text(self):
        sentences = _split_into_sentences("")
        assert len(sentences) == 1  # devuelve [""] en lugar de fallar

    def test_multiple_paragraphs(self):
        text = "Primer párrafo aquí.\n\nSegundo párrafo aquí."
        sentences = _split_into_sentences(text)
        assert len(sentences) == 2


class TestTokenizeSimple:
    def test_basic_tokenization(self):
        tokens = _tokenize_simple("Los CETES rinden 11.25%")
        assert tokens == ["Los", "CETES", "rinden", "11.25%"]

    def test_separates_commas(self):
        tokens = _tokenize_simple("Banxico, CNBV y BMV")
        assert "," in tokens
        assert "Banxico" in tokens

    def test_preserves_decimal_numbers(self):
        """11.25 no debe partirse en '11' y '25' por el punto decimal."""
        tokens = _tokenize_simple("La tasa es 11.25 por ciento")
        assert "11.25" in tokens

    def test_empty_sentence(self):
        assert _tokenize_simple("") == []

    def test_preserves_percentage_sign(self):
        tokens = _tokenize_simple("La tasa es 11.25%")
        assert "11.25%" in tokens
