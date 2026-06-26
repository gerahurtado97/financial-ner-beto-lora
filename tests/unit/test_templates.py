"""
tests/unit/test_templates.py — Tests de las plantillas de generación.

Verifica que cada plantilla produce tokens y tags alineados correctamente
en esquema BIO — esto es la pieza más crítica del proyecto, ya que un
error aquí contamina silenciosamente todo el dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from financial_ner.data.schema import validate_example
from financial_ner.data.templates import (
    _build,
    _tag_span,
    t_entidad_emitio_monto,
    t_entidad_recorte_tasa_plazo,
    t_entidad_verbo_instr_plazo,
    t_entidad_verbo_tasa,
    t_instr_plazo_entidad,
    t_instrumento_plazo,
    t_instrumento_plazo_tasa,
    t_monto_instr_tasa,
    t_negativo_sin_entidades,
    t_simple_instr_entidad,
)


class TestTagSpan:
    def test_single_token_entity(self):
        toks, tags = _tag_span("Banxico", "ENTIDAD")
        assert toks == ["Banxico"]
        assert tags == ["B-ENTIDAD"]

    def test_multi_token_entity(self):
        toks, tags = _tag_span("91 días", "PLAZO")
        assert toks == ["91", "días"]
        assert tags == ["B-PLAZO", "I-PLAZO"]

    def test_three_token_entity(self):
        toks, tags = _tag_span("25 puntos base", "TASA")
        assert toks == ["25", "puntos", "base"]
        assert tags == ["B-TASA", "I-TASA", "I-TASA"]

    def test_empty_string(self):
        toks, tags = _tag_span("", "INSTR")
        assert toks == []
        assert tags == []


class TestBuild:
    def test_mixed_entity_and_outside(self):
        tokens, tags = _build(("Los", None), ("CETES", "INSTR"))
        assert tokens == ["Los", "CETES"]
        assert tags == ["O", "B-INSTR"]

    def test_skips_empty_spans(self):
        tokens, tags = _build(("", None), ("Banxico", "ENTIDAD"))
        assert tokens == ["Banxico"]
        assert tags == ["B-ENTIDAD"]

    def test_multiword_outside_all_o(self):
        tokens, tags = _build(("el clima estuvo", None),)
        assert tokens == ["el", "clima", "estuvo"]
        assert tags == ["O", "O", "O"]


class TestTemplates:
    """Cada plantilla debe producir tokens/tags alineados y válidos en BIO."""

    def test_instrumento_plazo(self):
        toks, tags = t_instrumento_plazo("CETES", "91 días")
        assert len(toks) == len(tags)
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert "B-INSTR" in tags
        assert "B-PLAZO" in tags

    def test_instrumento_plazo_tasa(self):
        toks, tags = t_instrumento_plazo_tasa("TIIE", "28 días", "11.25%")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert tags.count("B-INSTR") == 1
        assert tags.count("B-PLAZO") == 1
        assert tags.count("B-TASA") == 1

    def test_entidad_verbo_tasa(self):
        toks, tags = t_entidad_verbo_tasa("Banxico", "elevó", "25 puntos base")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert toks[0] == "Banxico"
        assert tags[0] == "B-ENTIDAD"

    def test_entidad_verbo_instr_plazo(self):
        toks, tags = t_entidad_verbo_instr_plazo(
            "Banxico", "mantuvo", "TIIE", "28 días"
        )
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert "B-ENTIDAD" in tags
        assert "B-INSTR" in tags
        assert "B-PLAZO" in tags

    def test_entidad_emitio_monto(self):
        toks, tags = t_entidad_emitio_monto("Hacienda", "100 millones de pesos", "CETES")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        # "100 millones de pesos" = 4 tokens, 1 B + 3 I
        assert tags.count("B-MONTO") == 1
        assert tags.count("I-MONTO") == 3

    def test_instr_plazo_entidad(self):
        toks, tags = t_instr_plazo_entidad("swap", "91 días", "CNBV")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert "B-INSTR" in tags
        assert "B-ENTIDAD" in tags

    def test_monto_instr_tasa(self):
        toks, tags = t_monto_instr_tasa("50 mdd", "BONDES", "9.75%")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert tags.count("B-MONTO") == 1
        assert tags.count("B-INSTR") == 1
        assert tags.count("B-TASA") == 1

    def test_entidad_recorte_tasa_plazo(self):
        toks, tags = t_entidad_recorte_tasa_plazo("Banxico", "50 puntos base", "91 días")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert "B-ENTIDAD" in tags
        assert "B-TASA" in tags
        assert "B-PLAZO" in tags

    def test_simple_instr_entidad(self):
        toks, tags = t_simple_instr_entidad("CETES", "CNBV")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)

    def test_negativo_sin_entidades(self):
        toks, tags = t_negativo_sin_entidades("El clima estuvo soleado")
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
        assert all(t == "O" for t in tags)

    @pytest.mark.parametrize("instr,plazo", [
        ("CETES", "91 días"),
        ("swap", "un mes"),
        ("forward", "tres años"),
        ("TIIE28", "182 días"),
    ])
    def test_instrumento_plazo_parametrize(self, instr, plazo):
        toks, tags = t_instrumento_plazo(instr, plazo)
        validate_example({"tokens": toks, "ner_tags": tags}, 0)
