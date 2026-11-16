# Saber 11: Predicting High Performance on Colombia's National Exam

Midterm project for [ML Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

**Saber 11** is the standardized exam that every student in Colombia takes at the end of high school.
It is run by ICFES, and its global score (0–500) is used for university admissions, scholarships,
and to rank schools. Results are strongly unequal across regions, school types and family backgrounds.

**Goal:** predict whether a student will reach **high performance** (top 25% of the global score)
using only information that is available **before** the exam: characteristics of the school
(public/private, urban/rural, schedule, bilingual...), the household (socioeconomic stratum,
parents' education, internet and computer at home...) and the student.

**How the model would be used:** a regional education office could score its students or schools before the exam and focus tutoring, scholarships or resources on those with the lowest predicted probability of high performance. The model also shows which factors are most associated with thegap, which helps explain where to act.

- **Target:** `alto_desempeno` = 1 if `punt_global` >= 75th percentile (287 points), 0 otherwise.
- **Type:** binary classification. **Metric:** ROC AUC.
- **Data leakage:** the subject scores (`punt_matematicas`, `punt_ingles`, etc.) are **not** used,
  because the global score is computed from them.

## Data

- Source: [Resultados únicos Saber 11](https://www.datos.gov.co/d/kgxf-xxbe), ICFES,
  Colombian Open Data Portal. License CC BY-SA 4.0.
- Period: 20224 (one exam period, ~530,000 students after removing duplicated rows).
- A random sample of 50,000 students is committed in `data/saber11_sample.csv`.

To download it again:

```bash
uv run python download_data.py --list-periods
uv run python download_data.py --periodo 20224
```

## EDA and model selection

All the work is in [`notebook.ipynb`](notebook.ipynb):

1. Cleaning: duplicates, missing values (unanswered household questions are kept as an `NA` category).
2. EDA: distribution of the global score, high-performance rate by stratum, school type, area, internet, parents' education.
3. Feature importance: mutual information with the target.
4. Models (trained on train, compared on validation, 60/20/20 split):

| Model | Tuned parameters | Best validation AUC |
|---|---|---|
| Logistic regression | `C` | 0.778 |
| Random forest | `max_depth`, `min_samples_leaf` | 0.771 |
| XGBoost | `learning_rate`, `max_depth` | 0.781 |

**Final model:** XGBoost (`learning_rate=0.3`, `max_depth=3`, 200 trees) — test AUC: 0.7767

Main findings: <!-- 2-3 bullets from your EDA conclusions -->

## Project structure

```
├── data/saber11_sample.csv   data sample
├── download_data.py          downloads the data and builds the sample
├── notebook.ipynb            cleaning, EDA, feature importance, model selection
├── train.py                  trains the final model -> model.bin
├── model.bin                 trained model (pipeline: DictVectorizer + model)
├── predict.py                web service (FastAPI)
├── test_predict.py           example request
├── pyproject.toml, uv.lock   dependencies
└── Dockerfile
```

## How to run

### 1. Environment

The project uses [uv](https://docs.astral.sh/uv/) for dependencies and the virtual environment.

```bash
pip install uv       # if you don't have it
uv sync              # creates .venv and installs the exact versions from uv.lock
```

To use the environment in Jupyter/VS Code, select the kernel `.venv/bin/python`
(`.venv\Scripts\python.exe` on Windows). To activate it in a terminal:

```bash
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

### 2. Train the model

```bash
uv run python train.py
```

### 3. Run the web service locally

```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```

In another terminal:

```bash
uv run python test_predict.py
```

Example response:

```json
{"probabilidad_alto_desempeno": 0.0969, "alto_desempeno": false}
```

Interactive docs are available at http://localhost:9696/docs

### 4. Docker

```bash
docker build -t saber11-midterm .
docker run --rm -p 9696:9696 saber11-midterm
```

Then test it with `uv run python test_predict.py`.
