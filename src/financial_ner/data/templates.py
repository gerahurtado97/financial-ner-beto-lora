"""
templates.py — Plantillas de oraciones para generar el dataset sintético.

Cada plantilla es una función que recibe valores de entidades y devuelve
una tupla (tokens, ner_tags) ya alineada en esquema BIO.

Diseño: construir el texto entidad por entidad (no con regex sobre la
oración completa) garantiza que el etiquetado sea exacto desde el origen,
sin ambigüedad de qué palabra pertenece a qué entidad.
"""

from __future__ import annotations

from collections.abc import Callable

# Tipo de plantilla: función que devuelve (tokens, ner_tags)
TemplateFn = Callable[..., tuple[list[str], list[str]]]


def _tag_span(text: str, label: str) -> tuple[list[str], list[str]]:
    """
    Tokeniza una entidad (split simple por espacio) y genera sus tags BIO.

    Ejemplo: _tag_span("91 días", "PLAZO")
        → (["91", "días"], ["B-PLAZO", "I-PLAZO"])
    """
    toks = text.split()
    if not toks:
        return [], []
    tags = [f"B-{label}"] + [f"I-{label}"] * (len(toks) - 1)
    return toks, tags


def _build(*spans: tuple[str, str | None]) -> tuple[list[str], list[str]]:
    """
    Construye tokens y tags a partir de una secuencia de (texto, label_o_None).

    label=None significa texto fuera de cualquier entidad → tag "O".

    Ejemplo:
        _build(("Los", None), ("CETES", "INSTR"), ("a", None), ("91 días", "PLAZO"))
    """
    all_tokens: list[str] = []
    all_tags: list[str] = []
    for text, label in spans:
        if not text:
            continue
        if label is None:
            for tok in text.split():
                all_tokens.append(tok)
                all_tags.append("O")
        else:
            toks, tags = _tag_span(text, label)
            all_tokens.extend(toks)
            all_tags.extend(tags)
    return all_tokens, all_tags


# ─────────────────────────────────────────────────────────────
# Plantillas — cada una cubre un patrón sintáctico distinto
# ─────────────────────────────────────────────────────────────

def t_instrumento_plazo(instr: str, plazo: str) -> tuple[list[str], list[str]]:
    """'Los CETES a 91 días ...'"""
    return _build(
        ("Los", None), (instr, "INSTR"), ("a", None), (plazo, "PLAZO"),
        ("rinden", None),
    )


def t_instrumento_plazo_tasa(instr: str, plazo: str, tasa: str) -> tuple[list[str], list[str]]:
    """'Los CETES a 91 días rinden 11.25 por ciento'"""
    return _build(
        ("Los", None), (instr, "INSTR"), ("a", None), (plazo, "PLAZO"),
        ("rinden", None), (tasa, "TASA"),
    )


def t_entidad_verbo_tasa(entidad: str, verbo: str, tasa: str) -> tuple[list[str], list[str]]:
    """'Banxico elevó la tasa en 25 puntos base'"""
    return _build(
        (entidad, "ENTIDAD"), (verbo, None), ("la tasa en", None), (tasa, "TASA"),
    )


def t_entidad_verbo_instr_plazo(
    entidad: str, verbo: str, instr: str, plazo: str
) -> tuple[list[str], list[str]]:
    """'Banxico mantuvo la TIIE a 28 días sin cambio'"""
    return _build(
        (entidad, "ENTIDAD"), (verbo, None), ("la", None), (instr, "INSTR"),
        ("a", None), (plazo, "PLAZO"), ("sin cambio", None),
    )


def t_entidad_emitio_monto(entidad: str, monto: str, instr: str) -> tuple[list[str], list[str]]:
    """'Hacienda colocó 100 millones de pesos en CETES'"""
    return _build(
        (entidad, "ENTIDAD"), ("colocó", None), (monto, "MONTO"),
        ("en", None), (instr, "INSTR"),
    )


def t_instr_plazo_entidad(instr: str, plazo: str, entidad: str) -> tuple[list[str], list[str]]:
    """'El instrumento swap a 91 días fue registrado por la CNBV'"""
    return _build(
        ("El instrumento", None), (instr, "INSTR"), ("a", None), (plazo, "PLAZO"),
        ("fue registrado por la", None), (entidad, "ENTIDAD"),
    )


def t_monto_instr_tasa(monto: str, instr: str, tasa: str) -> tuple[list[str], list[str]]:
    """'Se colocaron 50 mdd en BONDES a una tasa de 9.75%'"""
    return _build(
        ("Se colocaron", None), (monto, "MONTO"), ("en", None), (instr, "INSTR"),
        ("a una tasa de", None), (tasa, "TASA"),
    )


def t_entidad_recorte_tasa_plazo(
    entidad: str, tasa: str, plazo: str
) -> tuple[list[str], list[str]]:
    """'La Junta de Gobierno de Banxico anunció un recorte de 50 puntos base para el plazo de 91 días'"""
    return _build(
        ("La Junta de Gobierno de", None), (entidad, "ENTIDAD"),
        ("anunció un recorte de", None), (tasa, "TASA"),
        ("para el plazo de", None), (plazo, "PLAZO"),
    )


def t_simple_instr_entidad(instr: str, entidad: str) -> tuple[list[str], list[str]]:
    """'Los CETES son regulados por la CNBV'"""
    return _build(
        ("Los", None), (instr, "INSTR"), ("son regulados por la", None), (entidad, "ENTIDAD"),
    )


def t_negativo_sin_entidades(frase: str) -> tuple[list[str], list[str]]:
    """Oraciones SIN ninguna entidad financiera — críticas para que el modelo
    no sobre-prediga entidades en texto genérico."""
    return _build((frase, None),)


# Lista de todas las plantillas con sus "arity" (cuántos argumentos reciben)
# usada por el generador para saber qué vocabularios pedir
ALL_TEMPLATES: list[TemplateFn] = [
    t_instrumento_plazo,
    t_instrumento_plazo_tasa,
    t_entidad_verbo_tasa,
    t_entidad_verbo_instr_plazo,
    t_entidad_emitio_monto,
    t_instr_plazo_entidad,
    t_monto_instr_tasa,
    t_entidad_recorte_tasa_plazo,
    t_simple_instr_entidad,
]

# Oraciones negativas (sin entidades) — ayudan a reducir falsos positivos
FRASES_NEGATIVAS = [
    "El clima estuvo soleado durante todo el fin de semana",
    "La reunión se llevará a cabo el próximo lunes por la tarde",
    "Los empleados asistieron a la capacitación anual",
    "El restaurante abrió sus puertas a las nueve de la mañana",
    "La conferencia tuvo una excelente asistencia este año",
    "El equipo de fútbol ganó el partido en tiempo extra",
    "La biblioteca permanecerá cerrada por mantenimiento",
    "Los estudiantes presentaron sus proyectos finales ayer",
    "El vuelo se retrasó por las condiciones climáticas",
    "La nueva película se estrena el próximo viernes",
]
