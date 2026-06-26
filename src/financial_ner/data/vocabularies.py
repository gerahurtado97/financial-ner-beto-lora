"""
vocabularies.py — Vocabularios realistas para generar el dataset sintético.

Cubre instrumentos y entidades mexicanas (Banxico, CETES, TIIE) y
términos genéricos/internacionales (swap, forward, Fed) según lo
especificado en el enunciado del proyecto.

Estos vocabularios alimentan el generador de plantillas en generate_dataset.py.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# INSTR — Instrumentos financieros
# ─────────────────────────────────────────────────────────────
INSTRUMENTOS_MX = [
    "CETES", "BONDES", "BONDES D", "UDIBONOS", "TIIE",
    "TIIE28", "TIIE91", "Bonos M", "Pagaré con Rendimiento Liquidable al Vencimiento",
    "PRLV", "Certificados Bursátiles", "Papel Comercial",
]

INSTRUMENTOS_INTL = [
    "swap", "forward", "futuro", "opción call", "opción put",
    "bono soberano", "eurobono", "credit default swap", "CDS",
    "repo", "cross-currency swap",
]

INSTRUMENTOS = INSTRUMENTOS_MX + INSTRUMENTOS_INTL

# ─────────────────────────────────────────────────────────────
# ENTIDAD — Entidades e instituciones
# ─────────────────────────────────────────────────────────────
ENTIDADES_MX = [
    "Banxico", "Banco de México", "CNBV",
    "Comisión Nacional Bancaria y de Valores", "BMV",
    "Bolsa Mexicana de Valores", "BIVA",
    "Bolsa Institucional de Valores", "Hacienda",
    "Secretaría de Hacienda y Crédito Público", "SHCP",
    "IPAB", "Banco de México y la Junta de Gobierno",
]

ENTIDADES_INTL = [
    "Fed", "Reserva Federal", "Banco Central Europeo", "BCE",
    "FMI", "Fondo Monetario Internacional", "Banco Mundial",
    "JP Morgan", "Goldman Sachs", "Moody's", "Standard & Poor's",
]

ENTIDADES = ENTIDADES_MX + ENTIDADES_INTL

# ─────────────────────────────────────────────────────────────
# PLAZO — Plazos y vencimientos (sin la preposición "a")
# ─────────────────────────────────────────────────────────────
PLAZOS = [
    "un día", "siete días", "28 días", "91 días", "182 días",
    "364 días", "un mes", "tres meses", "seis meses",
    "un año", "tres años", "cinco años", "diez años",
    "veinte años", "30 años",
]

# ─────────────────────────────────────────────────────────────
# TASA — Tasas y porcentajes (incluye "X puntos base" como una sola entidad)
# ─────────────────────────────────────────────────────────────
TASAS = [
    "11.25%", "10.50%", "9.75%", "8.25%", "6.00%",
    "11.25 por ciento", "diez por ciento", "ocho punto cinco por ciento",
    "25 puntos base", "50 puntos base", "75 puntos base",
    "100 puntos base", "quince puntos base",
]

# ─────────────────────────────────────────────────────────────
# MONTO — Cantidades monetarias (la frase completa es una sola entidad)
# ─────────────────────────────────────────────────────────────
MONTOS = [
    "100 millones de pesos", "50 mdd", "200 millones de dólares",
    "1,000 UDIs", "500 millones de pesos", "10 mil millones de pesos",
    "2.5 millones de dólares", "300 mdp", "75 mdd",
    "un millón de pesos", "cinco mil millones de pesos",
]

# ─────────────────────────────────────────────────────────────
# Verbos y conectores comunes en comunicados financieros
# (contexto "O" — fuera de cualquier entidad)
# ─────────────────────────────────────────────────────────────
VERBOS_POLITICA = [
    "elevó", "incrementó", "redujo", "mantuvo", "ajustó",
    "decidió mantener", "votó por elevar", "anunció un recorte de",
    "subió", "bajó", "modificó",
]

CONECTORES_PLAZO = ["a"]  # se omite en la entidad final, pero aparece en el texto
