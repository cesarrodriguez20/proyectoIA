# fetch_api.py

import requests
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# ─────────────────────────────────────
# DEPARTAMENTOS + RADIOS PRECISOS
# ─────────────────────────────────────

zonas = {

    "santander": {
        "lat": 6.83,
        "lon": -73.12,
        "radio": 300
    },

    "tolima": {
        "lat": 4.44,
        "lon": -75.24,
        "radio": 300
    },

    "risaralda": {
        "lat": 4.81,
        "lon": -75.69,
        "radio": 300
    },

    "narino": {
        "lat": 1.21,
        "lon": -77.28,
        "radio": 300
    },

    "choco": {
        "lat": 5.69,
        "lon": -76.65,
        "radio": 300
    },

    "antioquia": {
        "lat": 6.25,
        "lon": -75.56,
        "radio": 300
    },

    "cauca": {
        "lat": 2.44,
        "lon": -76.61,
        "radio": 300
    }
}

# ─────────────────────────────────────
# DESCARGA USGS
# ─────────────────────────────────────

def obtener(lat, lon, nombre, radio):

    url = (
        "https://earthquake.usgs.gov/"
        "fdsnws/event/1/query"
    )

    params = {

        "format":
            "geojson",

        "starttime":
            "2012-01-01",

        "endtime":
            "2026-12-31",

        "latitude":
            lat,

        "longitude":
            lon,

        "maxradiuskm":
            radio
    }

    try:

        r = requests.get(
            url,
            params=params,
            timeout=30
        )

        data = r.json()

    except Exception as e:

        print(
            f"❌ Error en {nombre}: {e}"
        )

        return

    filas = []

    for e in data.get("features", []):

        try:

            p = e["properties"]

            g = e["geometry"]

            mag = p["mag"]

            if mag is None or mag <= 0:
                continue

            energia = (
                10
                **
                (1.5 * mag + 4.8)
            )

            filas.append({

                "date":
                    p["time"],

                "magnitude":
                    mag,

                "depth":
                    g["coordinates"][2],

                "latitude":
                    g["coordinates"][1],

                "longitude":
                    g["coordinates"][0],

                "energia":
                    energia,

                "zona":
                    nombre
            })

        except:
            continue

    if not filas:

        print(
            f"⚠️ {nombre}: sin datos"
        )

        return

    df = pd.DataFrame(filas)

    df["date"] = pd.to_datetime(
        df["date"],
        unit="ms"
    )

    df.to_csv(
        f"data/{nombre}.csv",
        index=False
    )

    print(
        f"✅ {nombre}: "
        f"{len(df)} registros"
    )

# ─────────────────────────────────────
# EJECUCIÓN
# ─────────────────────────────────────

for nombre, info in zonas.items():

    obtener(

        info["lat"],
        info["lon"],
        nombre,
        info["radio"]
    )

print("\n🌍 DESCARGA FINALIZADA")