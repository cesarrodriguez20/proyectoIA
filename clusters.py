# clusters.py

import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────
# CARGAR DATASET
# ─────────────────────────────────────

df = pd.read_csv(
    "data/dataset_limpio.csv"
)

# ─────────────────────────────────────
# VALIDAR
# ─────────────────────────────────────

if len(df) == 0:

    print("❌ dataset vacío")
    exit()

# ─────────────────────────────────────
# FILTRAR EVENTOS ÚTILES
# ─────────────────────────────────────

df = df[
    df["magnitude"] > 1
]

# ─────────────────────────────────────
# RESULTADOS
# ─────────────────────────────────────

resultados = []

# ─────────────────────────────────────
# ANALIZAR POR DEPARTAMENTO
# ─────────────────────────────────────

for zona in df["zona"].unique():

    print("\n" + "="*50)

    print(
        f"🌍 ANALIZANDO: "
        f"{zona.upper()}"
    )

    print("="*50)

    # ─────────────────────────────────
    # FILTRAR ZONA
    # ─────────────────────────────────

    df_z = (

        df[df["zona"] == zona]

        .copy()
    )

    # ─────────────────────────────────
    # VALIDAR MÍNIMO
    # ─────────────────────────────────

    if len(df_z) < 10:

        print(
            "⚠️ muy pocos eventos"
        )

        continue

    # ─────────────────────────────────
    # VARIABLES GEO
    # ─────────────────────────────────

    X = df_z[[

        "latitude",
        "longitude"

    ]].values

    # ─────────────────────────────────
    # ESCALAR
    # ─────────────────────────────────

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ─────────────────────────────────
    # DBSCAN
    # ─────────────────────────────────
    # eps:
    # menor = más precisión
    # mayor = clusters grandes
    # ─────────────────────────────────

    modelo = DBSCAN(

        eps=0.28,

        min_samples=6

    )

    clusters = modelo.fit_predict(
        X_scaled
    )

    df_z["cluster"] = clusters

    # ─────────────────────────────────
    # ELIMINAR RUIDO
    # ─────────────────────────────────

    df_clusters = df_z[
        df_z["cluster"] != -1
    ]

    if len(df_clusters) == 0:

        print(
            "⚠️ sin clusters detectados"
        )

        continue

    # ─────────────────────────────────
    # CLUSTER PRINCIPAL
    # ─────────────────────────────────

    cluster_top = (

        df_clusters

        .groupby("cluster")

        .agg({

            "energia": "sum",

            "magnitude": "count"

        })

        .sort_values(
            "energia",
            ascending=False
        )

        .index[0]
    )

    hotspot = df_clusters[
        df_clusters["cluster"] == cluster_top
    ]

    # ─────────────────────────────────
    # CENTRO DEL HOTSPOT
    # ─────────────────────────────────

    lat = hotspot["latitude"].mean()

    lon = hotspot["longitude"].mean()

    # ─────────────────────────────────
    # ENERGÍA
    # ─────────────────────────────────

    energia = hotspot["energia"].sum()

    eventos = len(hotspot)

    magnitud = hotspot["magnitude"].max()

    # ─────────────────────────────────
    # DIRECCIÓN MIGRACIÓN
    # ─────────────────────────────────

    hotspot = hotspot.sort_values(
        "date"
    )

    lat_ini = hotspot.iloc[0]["latitude"]
    lon_ini = hotspot.iloc[0]["longitude"]

    lat_fin = hotspot.iloc[-1]["latitude"]
    lon_fin = hotspot.iloc[-1]["longitude"]

    dir_lat = (
        "Norte"
        if lat_fin > lat_ini
        else "Sur"
    )

    dir_lon = (
        "Este"
        if lon_fin > lon_ini
        else "Oeste"
    )

    direccion = (
        f"{dir_lat}-{dir_lon}"
    )

    # ─────────────────────────────────
    # GUARDAR RESULTADO
    # ─────────────────────────────────

    resultados.append({

        "zona": zona,

        "latitud_hotspot": lat,

        "longitud_hotspot": lon,

        "eventos_cluster": eventos,

        "energia_cluster": energia,

        "magnitud_max": magnitud,

        "direccion_migracion": direccion
    })

    # ─────────────────────────────────
    # RESUMEN
    # ─────────────────────────────────

    print(
        f"✅ hotspot:"
    )

    print(
        f"Lat: {lat:.4f}"
    )

    print(
        f"Lon: {lon:.4f}"
    )

    print(
        f"Eventos: {eventos}"
    )

    print(
        f"Energía: {energia:.2e}"
    )

    print(
        f"Migración: {direccion}"
    )

# ─────────────────────────────────────
# GUARDAR CSV
# ─────────────────────────────────────

df_out = pd.DataFrame(
    resultados
)

df_out.to_csv(

    "data/hotspots.csv",

    index=False
)

# ─────────────────────────────────────
# FINAL
# ─────────────────────────────────────

print("\n🌍 HOTSPOTS GENERADOS")

print(df_out)