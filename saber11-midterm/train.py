"""
Entrena el modelo final y lo guarda en model.bin

Uso:
    python train.py
"""

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBClassifier

DATA = "data/saber11_sample.csv"
MODEL_FILE = "model.bin"

# alto desempeno = estar en el 25% superior del puntaje global
QUANTILE = 0.75

# mismas candidatas del notebook (seccion 2). NUNCA incluir punt_* (fuga de datos)
CANDIDATAS = [
    "cole_naturaleza", "cole_area_ubicacion", "cole_jornada", "cole_bilingue",
    "cole_caracter", "cole_genero", "cole_calendario", "cole_depto_ubicacion",
    "estu_genero",
    "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
    "fami_tienelavadora", "fami_personashogar", "fami_cuartoshogar",
]

# TODO: aqui va lo que imprime la ultima celda del notebook
MODELO = 'xgboost'
PARAMS = {'learning_rate': 0.3, 'max_depth': 3}


def crear_modelo(nombre, params):
    if nombre == "logistica":
        return LogisticRegression(max_iter=1000, solver="lbfgs", **params)
    if nombre == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1, **params)
    return XGBClassifier(n_estimators=200, random_state=1, n_jobs=-1, **params)


def load_data():
    df = pd.read_csv(DATA, dtype=str).drop_duplicates()
    features = [c for c in CANDIDATAS if c in df.columns]
    df[features] = df[features].fillna("NA")

    df["punt_global"] = pd.to_numeric(df["punt_global"], errors="coerce")
    df = df[df["punt_global"].notna()].copy()

    umbral = df["punt_global"].quantile(QUANTILE)
    df["alto_desempeno"] = (df["punt_global"] >= umbral).astype(int)
    return df, features, umbral


def main():
    df, features, umbral = load_data()
    print(f"umbral de alto desempeno: {umbral}")
    print(f"modelo: {MODELO} {PARAMS}")

    # mismo split del notebook
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

    pipeline = make_pipeline(DictVectorizer(sparse=True), crear_modelo(MODELO, PARAMS))
    pipeline.fit(df_full_train[features].to_dict(orient="records"), df_full_train["alto_desempeno"])

    pred = pipeline.predict_proba(df_test[features].to_dict(orient="records"))[:, 1]
    print(f"AUC en test: {roc_auc_score(df_test['alto_desempeno'], pred):.4f}")

    with open(MODEL_FILE, "wb") as f_out:
        pickle.dump({"pipeline": pipeline, "features": features, "threshold": umbral}, f_out)
    print(f"modelo guardado en {MODEL_FILE}")


if __name__ == "__main__":
    main()
