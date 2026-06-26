"""
vocabularies.py — Vocabularios realistas para generar el dataset sintético.

Cubre instrumentos y entidades mexicanas (Banxico, CETES, TIIE) y
términos genéricos/internacionales (swap, forward, Fed) según lo
especificado en el enunciado del proyecto.

DISEÑO ANTI-MEMORIZACIÓN (corrección crítica):
    Si el mismo vocabulario aparece en train Y en val/test, el modelo
    puede alcanzar F1=1.0 memorizando palabras específicas ("CETES
    siempre es B-INSTR") en lugar de aprender el patrón sintáctico BIO.

    Para forzar generalización real, cada vocabulario se divide en
    pools DISJUNTOS — train usa un subconjunto de valores, val/test
    usan valores que el modelo NUNCA vio en entrenamiento. Así, un F1
    alto en val solo es posible si el modelo aprendió el patrón
    (posición, contexto, estructura BIO), no las palabras exactas.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# INSTR — Instrumentos financieros
# ─────────────────────────────────────────────────────────────
INSTRUMENTOS_MX_TRAIN = [
    "CETES", "TIIE", "TIIE28", "Bonos M", "Papel Comercial",
    "PRLV", "Pagaré con Rendimiento Liquidable al Vencimiento",
]
INSTRUMENTOS_MX_HELDOUT = [
    "BONDES", "BONDES D", "UDIBONOS", "TIIE91", "Certificados Bursátiles",
]

INSTRUMENTOS_INTL_TRAIN = [
    "swap", "forward", "futuro", "repo", "eurobono", "bono soberano",
]
INSTRUMENTOS_INTL_HELDOUT = [
    "opción call", "opción put", "credit default swap", "CDS",
    "cross-currency swap",
]

INSTRUMENTOS_TRAIN = INSTRUMENTOS_MX_TRAIN + INSTRUMENTOS_INTL_TRAIN
INSTRUMENTOS_HELDOUT = INSTRUMENTOS_MX_HELDOUT + INSTRUMENTOS_INTL_HELDOUT
INSTRUMENTOS = INSTRUMENTOS_TRAIN + INSTRUMENTOS_HELDOUT  # para EDA/notebooks

# ─────────────────────────────────────────────────────────────
# ENTIDAD — Entidades e instituciones
# ─────────────────────────────────────────────────────────────
ENTIDADES_MX_TRAIN = [
    "Banxico", "Banco de México", "CNBV", "BMV",
    "Bolsa Mexicana de Valores", "Hacienda",
]
ENTIDADES_MX_HELDOUT = [
    "Comisión Nacional Bancaria y de Valores", "BIVA",
    "Bolsa Institucional de Valores", "Secretaría de Hacienda y Crédito Público",
    "SHCP", "IPAB",
]

ENTIDADES_INTL_TRAIN = [
    "Fed", "Reserva Federal", "FMI", "Banco Mundial",
]
ENTIDADES_INTL_HELDOUT = [
    "Banco Central Europeo", "BCE", "Fondo Monetario Internacional",
    "JP Morgan", "Goldman Sachs", "Moody's", "Standard & Poor's",
]

ENTIDADES_TRAIN = ENTIDADES_MX_TRAIN + ENTIDADES_INTL_TRAIN
ENTIDADES_HELDOUT = ENTIDADES_MX_HELDOUT + ENTIDADES_INTL_HELDOUT
ENTIDADES = ENTIDADES_TRAIN + ENTIDADES_HELDOUT

# ─────────────────────────────────────────────────────────────
# PLAZO — Plazos y vencimientos (sin la preposición "a")
# ─────────────────────────────────────────────────────────────
PLAZOS_TRAIN = [
    "un día", "28 días", "91 días", "un mes", "tres meses",
    "un año", "cinco años", "diez años",
]
PLAZOS_HELDOUT = [
    "siete días", "182 días", "364 días", "seis meses",
    "tres años", "veinte años", "30 años",
]
PLAZOS = PLAZOS_TRAIN + PLAZOS_HELDOUT

# ─────────────────────────────────────────────────────────────
# TASA — Tasas y porcentajes
# ─────────────────────────────────────────────────────────────
TASAS_TRAIN = [
    "11.25%", "9.75%", "6.00%", "11.25 por ciento", "diez por ciento",
    "25 puntos base", "50 puntos base", "100 puntos base",
]
TASAS_HELDOUT = [
    "10.50%", "8.25%", "ocho punto cinco por ciento",
    "75 puntos base", "quince puntos base",
]
TASAS = TASAS_TRAIN + TASAS_HELDOUT

# ─────────────────────────────────────────────────────────────
# MONTO — Cantidades monetarias
# ─────────────────────────────────────────────────────────────
MONTOS_TRAIN = [
    "100 millones de pesos", "50 mdd", "500 millones de pesos",
    "10 mil millones de pesos", "un millón de pesos",
]
MONTOS_HELDOUT = [
    "200 millones de dólares", "1,000 UDIs", "2.5 millones de dólares",
    "300 mdp", "75 mdd", "cinco mil millones de pesos",
]
MONTOS = MONTOS_TRAIN + MONTOS_HELDOUT

# ─────────────────────────────────────────────────────────────
# Verbos y conectores comunes en comunicados financieros
# Estos NO son entidades — no necesitan separación train/heldout,
# pero igual variamos para robustecer el contexto "O"
# ─────────────────────────────────────────────────────────────
VERBOS_POLITICA = [
    "elevó", "incrementó", "redujo", "mantuvo", "ajustó",
    "decidió mantener", "votó por elevar", "anunció un recorte de",
    "subió", "bajó", "modificó",
]

CONECTORES_PLAZO = ["a"]  # se omite en la entidad final, pero aparece en el texto
