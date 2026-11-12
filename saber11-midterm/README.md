# Saber 11: predicción de alto desempeño

Proyecto de mitad de curso (midterm) del [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## El problema

<!-- TODO: 1-2 párrafos. Qué es Saber 11, por qué importa, quién usaría el modelo
(ej. secretarías de educación que quieren focalizar refuerzos antes del examen)
y qué decisión apoyaría. -->

**Variable objetivo:** `alto_desempeno` = 1 si `punt_global >= <UMBRAL>`. <!-- TODO: justificar el umbral -->

**Variables usadas:** solo información disponible antes del examen (colegio, hogar y estudiante).
Los puntajes por área se excluyen porque forman el puntaje global (fuga de datos).

## Datos

- Fuente: [Resultados únicos Saber 11](https://www.datos.gov.co/d/kgxf-xxbe), ICFES, Portal de Datos Abiertos de Colombia. Licencia CC BY-SA 4.0.
- Periodo usado: <!-- TODO -->
- Muestra aleatoria de 50.000 estudiantes, incluida en `data/saber11_sample.csv`.

Para volver a descargarla:

```bash
uv run python download_data.py --list-periods
uv run python download_data.py --periodo <PERIODO>
```

## Resultados

<!-- TODO: tabla con la AUC de cada modelo en validación y el modelo elegido -->

| Modelo | AUC validación |
|---|---|
| Regresión logística | |
| Random forest | |
| XGBoost | |

## Estructura

```
├── data/saber11_sample.csv   muestra de datos
├── download_data.py          descarga y muestreo
├── notebook.ipynb            limpieza, EDA, selección de modelos
├── train.py                  entrena el modelo final -> model.bin
├── predict.py                API con FastAPI
├── test_predict.py           petición de ejemplo
├── pyproject.toml / uv.lock  dependencias
└── Dockerfile
```

## Cómo ejecutarlo

### 1. Entorno

Requiere [uv](https://docs.astral.sh/uv/).

```bash
uv sync                 # crea .venv e instala todo
```

### 2. Entrenar

```bash
uv run python train.py
```

### 3. Servicio local

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

En otra terminal:

```bash
uv run python test_predict.py
```

### 4. Docker

```bash
docker build -t saber11-midterm .
docker run --rm -p 9696:9696 saber11-midterm
```

Y probar con `uv run python test_predict.py`.

## Despliegue en la nube

<!-- TODO (opcional, 2 puntos): URL del servicio o captura/video probándolo -->
