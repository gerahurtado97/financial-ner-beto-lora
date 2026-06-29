# NER Financiero con BETO + LoRA

Reconocimiento de Entidades Nombradas para texto financiero en español,
mediante fine-tuning eficiente (LoRA) de BETO (BERT en español).

**Proyecto Final Módulo 2 · Diplomado en AI & LLM for Financial Markets · ITAM**

---

## Descripción

Sistema de NER que extrae 5 tipos de entidades financieras de texto en español:

| Etiqueta | Descripción | Ejemplos |
|---|---|---|
| `INSTR` | Instrumento financiero | CETES, TIIE, BONDES, swap, forward |
| `MONTO` | Cantidad monetaria | 100 millones de pesos, 50 mdd |
| `PLAZO` | Plazo o vencimiento | 91 días, un año |
| `TASA` | Tasa o porcentaje | 11.25%, 25 puntos base |
| `ENTIDAD` | Entidad o institución | Banxico, CNBV, Fed, JP Morgan |

Implementado con:
- **Modelo base**: BETO (`dccuchile/bert-base-spanish-wwm-cased`)
- **Fine-tuning eficiente**: LoRA (Low-Rank Adaptation) sobre `query`/`value`
- **Dataset**: sintético, generado con plantillas y vocabulario MX + internacional
- **Evaluación**: seqeval a nivel de entidad (P/R/F1)

---

## Resultados

Comparación de 3 configuraciones LoRA en **test set** (vocabulario heldout,
nunca visto en entrenamiento):

| r | F1 | Precision | Recall | Params entrenables |
|---|---|---|---|---|
| 8 | 0.8030 | 0.7644 | 0.8457 | 0.277% |
| **16** | **0.8802** | **0.8622** | **0.8989** | 0.545% |
| 32 | 0.8571 | 0.8235 | 0.8936 | 1.076% |

**r=16 es la configuración óptima** — mejor F1 con un costo de parámetros
moderado. r=32 duplica los parámetros entrenables sin mejorar el resultado.

### F1 por tipo de entidad (r=16, test set)

| Entidad | Precision | Recall | F1 |
|---|---|---|---|
| MONTO | 1.000 | 1.000 | 1.000 |
| TASA | 1.000 | 1.000 | 1.000 |
| PLAZO | 0.951 | 1.000 | 0.975 |
| INSTR | 0.821 | 0.852 | 0.836 |
| ENTIDAD | 0.706 | 0.766 | 0.735 |

---

## Instalación

```bash
git clone https://github.com/gerahurtado97/financial-ner-beto-lora.git
cd financial-ner-beto-lora

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install pip-tools
pip install -r requirements.lock
pip install -r requirements-dev.lock
pip install -e ".[dev]"

# IMPORTANTE: torch llega como dependencia transitiva de accelerate/peft,
# y pip-compile siempre resuelve esa entrada desde PyPI estándar (CPU-only)
# porque "torch" existe con ese mismo nombre tanto en PyPI como en el
# índice de PyTorch — no hay forma de que el lockfile fije la build CUDA
# automáticamente. Por eso se reinstala torch explícitamente después,
# sobre lo que ya quedó instalado:
pip install torch --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-deps
# (ajusta cu121 a tu versión de CUDA si es distinta — ver
#  https://pytorch.org/get-started/locally/)
# --no-deps evita que pip intente resincronizar accelerate/peft/transformers,
# que ya quedaron correctamente instalados por el paso anterior

pre-commit install
cp .env.example .env

# Verificar que torch detecta la GPU correctamente:
python -c "import torch; print(torch.cuda.is_available())"
# Debe imprimir True
```

---

## Cómo ejecutar

```bash
# Etapa 1: generar dataset sintético (train con vocab_pool, val/test con heldout)
python scripts/generate_dataset.py

# Etapa 2: validar splits
python scripts/prepare_data.py

# Etapa 3: entrenar las 3 configuraciones LoRA
python scripts/train.py --lora-r 8  --lora-alpha 16
python scripts/train.py --lora-r 16 --lora-alpha 32
python scripts/train.py --lora-r 32 --lora-alpha 64

# Etapa 4: evaluación final en test set
python scripts/evaluate.py

# Inferencia — el entregable principal
python inference.py docs/ejemplos_txt/ejemplo1.txt
python inference.py ruta/a/tu/archivo.txt
```

### Uso de `inference.py`

```bash
python inference.py archivo.txt
```

El archivo debe contener texto financiero en español. Salida — una
entidad por línea:

```
CETES INSTR
91 días PLAZO
11.25 por ciento TASA
Banxico ENTIDAD
```

Si no se encuentran entidades, imprime `Sin entidades detectadas.`

Por defecto usa el modelo `r=16` (`models/ner_lora_r16/lora_weights`),
la configuración óptima encontrada en la comparación.

### Uso con Docker (reproducibilidad garantizada)

Al probar el proyecto en una segunda laptop se detectaron problemas de
reproducibilidad: Python 3.13 no tiene wheel precompilado para
`numpy==1.26.4` (fijado en el lockfile) y requiere un compilador C/C++
no disponible en esa máquina. Se agregó un Dockerfile que fija
Python 3.11 y todas las dependencias exactas, independiente del
sistema anfitrión.

```bash
# Construir la imagen (incluye el modelo r=16, el óptimo)
docker build -t financial-ner-lora .

# Correr con el ejemplo por defecto
docker run financial-ner-lora

# Correr con tu propio archivo .txt (monta un volumen local)
docker run -v /ruta/absoluta/a/tu/carpeta:/data financial-ner-lora \
    python inference.py /data/tu_archivo.txt
```

Esto elimina cualquier dependencia de la versión de Python instalada,
compilador C/C++, o políticas de seguridad de Windows (como el bloqueo
de DLL de `safetensors` observado en una laptop corporativa) — el
contenedor siempre corre exactamente el mismo entorno.

---

## La pieza técnica central: alineación de subtokens

El dataset etiqueta a nivel de **palabra**, pero BETO usa WordPiece y
puede dividir una palabra en varios subtokens:

```
Palabra:    UDIBONOS
Etiqueta:   B-INSTR
Subtokens:  UDI ##BON ##OS        (3 subtokens para 1 palabra)
```

Sin realinear, el modelo recibiría más subtokens que etiquetas. La
solución (`src/financial_ner/data/alignment.py`):

1. El **primer subtoken** de cada palabra recibe la etiqueta original
2. Los subtokens **siguientes** de la misma palabra reciben `-100`
   (índice que PyTorch `CrossEntropyLoss` ignora)
3. Los tokens especiales (`[CLS]`, `[SEP]`, `[PAD]`) también reciben `-100`

```
Subtokens:   [CLS]  UDI    ##BON  ##OS   [SEP]
Etiquetas:   -100   B-INSTR -100   -100   -100
```

Verificado con 12 tests usando el tokenizador real de BETO
(`tests/unit/test_alignment.py`).

---

## Corrección crítica: vocabulario disjunto train/heldout

**Primer experimento (sin esta corrección):** el mismo vocabulario
(`CETES`, `Banxico`, etc.) aparecía en train y en val/test. El modelo
alcanzó `eval_f1=1.0` desde la época 4 — **memorización**, no
aprendizaje del patrón BIO.

**Solución:** `src/financial_ner/data/vocabularies.py` divide cada
categoría en dos pools disjuntos:

```python
INSTRUMENTOS_MX_TRAIN   = ["CETES", "TIIE", "Bonos M", ...]
INSTRUMENTOS_MX_HELDOUT = ["BONDES", "UDIBONOS", "TIIE91", ...]
```

`train.json` se genera con `vocab_pool="train"`; `val.json` y
`test.json` con `vocab_pool="heldout"` — el modelo nunca ve esas
palabras exactas en entrenamiento. Esto fuerza al modelo a aprender
el **patrón sintáctico** (posición, contexto) en lugar de memorizar
palabras específicas.

**Resultado tras la corrección:** F1 realista (0.80-0.88) en lugar
del falso 1.0 — evidencia de generalización real, verificada con
entidades heldout como `BONDES`, `UDIBONOS`, `JP Morgan` en los
ejemplos de `inference.py`.

---

## Estructura del proyecto

```
financial-ner-beto-lora/
├── src/financial_ner/
│   ├── data/
│   │   ├── vocabularies.py    <- Pools TRAIN/HELDOUT disjuntos por entidad
│   │   ├── templates.py       <- 9 plantillas sintácticas + negativas
│   │   ├── generate_dataset.py<- generate_train_val_test()
│   │   ├── schema.py          <- Validación BIO (alineación, etiquetas válidas)
│   │   ├── labels.py          <- Mapeo BIO <-> índices
│   │   ├── alignment.py       <- Alineación subtoken (pieza más evaluada)
│   │   └── split.py           <- (legacy, ya no usado tras el fix de vocab)
│   ├── model/
│   │   └── lora_model.py      <- BETO + LoRA, comparación r=8/16/32
│   └── evaluation/
│       ├── metrics.py         <- seqeval P/R/F1 (global + por entidad)
│       └── mlflow_logger.py   <- Tracking de los 3 runs LoRA
├── scripts/
│   ├── generate_dataset.py    <- Genera train/val/test con vocab disjunto
│   ├── prepare_data.py        <- Valida los splits generados
│   ├── train.py                <- Entrenamiento (--lora-r, --lora-alpha)
│   └── evaluate.py             <- Evaluación final en test set
├── inference.py                <- ENTREGABLE PRINCIPAL (raíz del proyecto)
├── Dockerfile                   <- Imagen reproducible (Python 3.11 fijo, CPU-only)
├── .dockerignore
├── docs/ejemplos_txt/           <- Archivos .txt de prueba para inference.py
├── notebooks/
│   └── eda_dataset.ipynb        <- Balance de clases, longitud, vocabulario
├── tests/unit/                  <- 94 tests
├── configs/model_config.yaml    <- Hiperparámetros (LoRA, training, labels)
├── requirements.lock            <- Lockfile con hashes (pip-compile)
├── dvc.yaml                     <- Pipeline: generate → train_r8/16/32 → evaluate
└── reporte.pdf                  <- Reporte técnico (2-3 páginas)
```

---

## Tests

```bash
pytest tests/ -v
# 94 tests: 75 sin red + 19 de inference.py (puras) 
# (12 de alignment.py requieren descargar BETO la primera vez)
```

| Archivo | Tests | Qué verifica |
|---|---|---|
| `test_schema.py` | 13 | Validación BIO: alineación, etiquetas válidas, I- sin B- |
| `test_templates.py` | 16 | Las 9 plantillas producen BIO correcto |
| `test_generate_dataset.py` | 16 | Generador, deduplicación, **vocabulario disjunto train/heldout** |
| `test_alignment.py` | 12 | Alineación de subtokens con BETO real |
| `test_labels.py` | 7 | Mapeo BIO ↔ índices |
| `test_split.py` | 5 | Split train/val/test (legacy) |
| `test_inference.py` | 19 | `extract_entities`, tokenización, split de oraciones |

---

## MLflow

```bash
mlflow ui --port 5000
```

Los 3 runs (`r=8`, `r=16`, `r=32`) están en el experimento
`financial-ner-beto-lora`, con métricas por época y F1 final por
tipo de entidad.

---

## Reproducibilidad

### Lockfile

```bash
pip-compile requirements.in --generate-hashes -o requirements.lock
pip-compile requirements-dev.in --generate-hashes -o requirements-dev.lock
```

### Pipeline DVC

```bash
dvc repro
```

No requiere acceso a ningún remote — el dataset es 100% sintético
y se genera localmente con `scripts/generate_dataset.py`.

---

