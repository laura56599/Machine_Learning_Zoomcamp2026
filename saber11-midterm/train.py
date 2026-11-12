"""
Entrena el modelo final y lo guarda en model.bin

Uso:
    python train.py
"""

import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

DATA = "data/saber11_sample.csv"
MODEL_FILE = "model.bin"

# TODO: ajustar despues del EDA
THRESHOLD = 300  # alto desempeno = punt_global >= THRESHOLD

# TODO: dejar solo las variables que elijas en el notebook
# NUNCA incluir punt_* (son parte del puntaje global -> fuga de datos)
CATEGORICAL = [
    "cole_naturaleza",
    "cole_area_ubicacion",
    "cole_jornada",
    "cole_bilingue",
    "cole_caracter",
    "cole_depto_ubicacion",
    "estu_genero",
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_personashogar",
    "fami_cuartoshogar",
]
NUMERICAL = []

# TODO: poner los mejores parametros que encuentres en el notebook
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.1,
    "random_state": 1,
    "n_jobs": -1,
}


def load_data():
    df = pd.read_csv(DATA, dtype=str)
    df["punt_global"] = pd.to_numeric(df["punt_global"])
    df["alto_desempeno"] = (df["punt_global"] >= THRESHOLD).astype(int)

    columnas = [c for c in CATEGORICAL + NUMERICAL if c in df.columns]
    faltan = set(CATEGORICAL + NUMERICAL) - set(columnas)
    if faltan:
        print("aviso: estas columnas no estan en el dataset:", sorted(faltan))

    df[columnas] = df[columnas].fillna("NA")
    for c in NUMERICAL:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df, columnas


def main():
    df, columnas = load_data()

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    pipeline = make_pipeline(DictVectorizer(sparse=True), XGBClassifier(**XGB_PARAMS))
    pipeline.fit(df_full_train[columnas].to_dict(orient="records"), df_full_train["alto_desempeno"])

    pred = pipeline.predict_proba(df_test[columnas].to_dict(orient="records"))[:, 1]
    print(f"AUC en test: {roc_auc_score(df_test['alto_desempeno'], pred):.3f}")

    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump({"pipeline": pipeline, "features": columnas, "threshold": THRESHOLD}, f_out)
    print(f"modelo guardado en {MODEL_FILE}")


if __name__ == "__main__":
    main()
