"""Envia un estudiante de ejemplo al servicio. Uso: python test_predict.py [url]"""

import sys

import requests

url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9696/predict"

# TODO: ajustar los valores a las categorias reales que veas en el EDA
estudiante = {
    "cole_naturaleza": "OFICIAL",
    "cole_area_ubicacion": "URBANO",
    "cole_jornada": "MAÑANA",
    "cole_bilingue": "N",
    "cole_caracter": "ACADÉMICO",
    "cole_depto_ubicacion": "BOGOTÁ",
    "estu_genero": "F",
    "fami_estratovivienda": "Estrato 3",
    "fami_educacionmadre": "Técnica o tecnológica completa",
    "fami_educacionpadre": "Secundaria (Bachillerato) completa",
    "fami_tieneinternet": "Si",
    "fami_tienecomputador": "Si",
    "fami_personashogar": "3 a 4",
    "fami_cuartoshogar": "Tres",
}

print(requests.post(url, json=estudiante, timeout=30).json())
