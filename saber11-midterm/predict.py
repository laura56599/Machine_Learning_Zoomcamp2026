"""
Servicio web que predice la probabilidad de alto desempeno en Saber 11

Uso local:
    uvicorn predict:app --host 0.0.0.0 --port 9696
"""

import pickle

from fastapi import FastAPI

with open("model.bin", "rb") as f_in:
    artefacto = pickle.load(f_in)

pipeline = artefacto["pipeline"]
features = artefacto["features"]

app = FastAPI(title="saber11-alto-desempeno")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(estudiante: dict):
    # las variables que no lleguen se tratan como 'NA', igual que en el entrenamiento
    x = {f: str(estudiante.get(f, "NA")) for f in features}
    prob = float(pipeline.predict_proba([x])[0, 1])
    return {
        "probabilidad_alto_desempeno": round(prob, 4),
        "alto_desempeno": prob >= 0.5,
    }
