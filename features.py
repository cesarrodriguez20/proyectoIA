# features.py

import pandas as pd
import numpy as np

# ─────────────────────────────────────
# CARGAR DATASET LIMPIO
# ─────────────────────────────────────

df = pd.read_csv(
    "data/dataset_limpio.csv"
)

df["date"] = pd.to_datetime(
    df["date"]
)

# ─────────────────────────────────────
# AGRUPACIÓN DIARIA
# PREDICCIÓN PRÓXIMAS 24H
# ─────────────────────────────────────

df["fecha"] = (
    df["date"]
    .dt.date
)

diario = df.groupby(

    ["zona", "fecha"]

).agg({

    "magnitude":
        ["count", "mean", "max"],

    "depth":
        ["mean", "std", "max"],

    "energia":
        ["sum", "max"],

    "delta_t":
        "mean",

    "mag_7d":
        "mean",

    "energia_7d":
        "mean"

}).reset_index()

# ─────────────────────────────────────
# RENOMBRAR COLUMNAS
# ─────────────────────────────────────

diario.columns = [

    "zona",
    "fecha",

    "conteo",

    "mag_promedio",
    "mag_max",

    "depth_promedio",
    "depth_std",
    "depth_max",

    "energia_total",
    "energia_max",

    "delta_t",

    "mag_7d",
    "energia_7d"
]

# ─────────────────────────────────────
# DELTA ENERGÍA
# ─────────────────────────────────────

diario["energia_prev"] = (

    diario.groupby("zona")
    ["energia_total"]

    .shift(1)
)

diario["delta_energia"] = (

    diario["energia_total"]
    -
    diario["energia_prev"]
)

# ─────────────────────────────────────
# PESOS TECTÓNICOS
# ─────────────────────────────────────

pesos = {

    "santander": 1.5,

    "tolima": 1.3,

    "risaralda": 1.2,

    "narino": 1.6,

    "choco": 1.7,

    "antioquia": 1.1,

    "cauca": 1.5
}

diario["peso_tectonico"] = (
    diario["zona"].map(pesos)
)

diario["riesgo_tectonico"] = (

    diario["energia_total"]
    *
    diario["peso_tectonico"]
)

# ─────────────────────────────────────
# VARIABLES LAG
# ─────────────────────────────────────

for i in range(1, 4):

    diario[f"lag_{i}"] = (

        diario.groupby("zona")
        ["conteo"]

        .shift(i)
    )

# ─────────────────────────────────────
# B-VALUE
# ─────────────────────────────────────

diario["b_value"] = (

    diario.groupby("zona")
    ["mag_promedio"]

    .rolling(
        30,
        min_periods=2
    )

    .apply(

        lambda x:

        np.log10(np.e)
        /
        (
            x.mean()
            -
            x.min()
            +
            0.001
        )
    )

    .reset_index(level=0, drop=True)
)

# ─────────────────────────────────────
# ACELERACIÓN SÍSMICA
# ─────────────────────────────────────

diario["aceleracion"] = (

    diario.groupby("zona")
    ["conteo"]

    .diff()
)

# ─────────────────────────────────────
# ENERGÍA ACUMULADA
# ─────────────────────────────────────

diario["energia_30d"] = (

    diario.groupby("zona")
    ["energia_total"]

    .rolling(
        30,
        min_periods=1
    )

    .sum()

    .reset_index(level=0, drop=True)
)

# ─────────────────────────────────────
# TARGET IA
# ACTIVIDAD SÍSMICA PRÓXIMAS 24H
# ─────────────────────────────────────

diario["target"] = (

    diario.groupby("zona")
    ["conteo"]

    .shift(-1)

    > 3

).astype(int)

# ─────────────────────────────────────
# LIMPIEZA
# ─────────────────────────────────────

diario = diario.replace(
    [np.inf, -np.inf],
    np.nan
)

diario = diario.fillna(0)

# ─────────────────────────────────────
# GUARDAR FEATURES
# ─────────────────────────────────────

diario.to_csv(

    "data/features_para_ia.csv",

    index=False
)

# ─────────────────────────────────────
# RESUMEN
# ─────────────────────────────────────

print("\n✅ FEATURES GENERADAS")

print(diario.shape)

print("\n📊 TARGETS POSITIVOS")

print(

    diario.groupby("zona")["target"]
    .sum()
)

print("\n📈 MUESTRAS POR ZONA")

print(

    diario.groupby("zona")
    .size()
)