"""
Descarga los resultados de Saber 11 desde datos.gov.co (ICFES)
y guarda una muestra en data/saber11_sample.csv

Fuente: "Resultados unicos Saber 11" - https://www.datos.gov.co/d/kgxf-xxbe
Licencia: CC BY-SA 4.0

Uso:
    python download_data.py --inspect              # ver columnas disponibles
    python download_data.py --list-periods         # ver periodos y cuantos registros tiene cada uno
    python download_data.py --periodo 20224        # descargar un periodo y guardar la muestra
"""

import argparse
import io
import sys
import time

import pandas as pd
import requests

BASE_URL = "https://www.datos.gov.co/resource/kgxf-xxbe.csv"
PAGE_SIZE = 50_000
SAMPLE_SIZE = 50_000
RANDOM_STATE = 1
OUTPUT = "data/saber11_sample.csv"


def get_csv(params):
    for intento in range(3):
        try:
            r = requests.get(BASE_URL, params=params, timeout=120)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        except requests.RequestException as e:
            print(f"  error ({e}), reintentando...")
            time.sleep(5 * (intento + 1))
    sys.exit("No se pudo descargar. Revisa la conexion o abre la URL en el navegador.")


def inspect():
    df = get_csv({"$limit": 5})
    print(f"{len(df.columns)} columnas:\n")
    for col in df.columns:
        print(f"  {col:35s} ej: {df[col].iloc[0]}")


def list_periods():
    df = get_csv({"$select": "periodo, count(*) AS n", "$group": "periodo", "$order": "periodo"})
    print(df.to_string(index=False))


def download(periodo):
    paginas = []
    offset = 0
    while True:
        print(f"descargando filas {offset:,} - {offset + PAGE_SIZE:,} ...")
        page = get_csv({
            "$where": f"periodo='{periodo}'",
            "$order": ":id",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        })
        if page.empty:
            break
        paginas.append(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    if not paginas:
        sys.exit(f"No hay datos para periodo={periodo}. Usa --list-periods para ver los validos.")

    df = pd.concat(paginas, ignore_index=True)
    print(f"\nfilas descargadas: {len(df):,}")

    # el portal puede traer filas repetidas
    if "estu_consecutivo" in df.columns:
        df = df.drop_duplicates(subset="estu_consecutivo")
    else:
        df = df.drop_duplicates()
    print(f"filas sin duplicados: {len(df):,}")

    # solo estudiantes con puntaje global
    df = df[pd.to_numeric(df["punt_global"], errors="coerce").notna()]

    if len(df) > SAMPLE_SIZE:
        df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    df.to_csv(OUTPUT, index=False)
    print(f"muestra guardada en {OUTPUT}: {len(df):,} filas, {len(df.columns)} columnas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--list-periods", action="store_true")
    parser.add_argument("--periodo", type=str)
    args = parser.parse_args()

    if args.inspect:
        inspect()
    elif args.list_periods:
        list_periods()
    elif args.periodo:
        download(args.periodo)
    else:
        parser.print_help()
