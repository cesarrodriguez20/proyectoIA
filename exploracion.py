# exploracion.py

import pandas as pd
import numpy as np
import os

archivos = [

    "santander",
    "tolima",
    "risaralda",
    "narino",
    "choco",
    "antioquia",
    "cauca"
]

dfs = []

for archivo in archivos:

    ruta = f"data/{archivo}.csv"

    if os.path.exists(ruta):

        dfs.append(
            pd.read_csv(ruta)
        )

df = pd.concat(
    dfs,
    ignore_index=True
)

df["date"] = pd.to_datetime(
    df["date"]
)

df = df.dropna()

df = df[
    (df["magnitude"] > 0)
    &
    (df["depth"] >= 0)
]

df = df.sort_values(
    ["zona", "date"]
)

# ─────────────────────────────────────
# FEATURES TEMPORALES
# ─────────────────────────────────────

df["delta_t"] = (

    df.groupby("zona")["date"]

    .diff()

    .dt.total_seconds()
)

df["mag_7d"] = (

    df.groupby("zona")["magnitude"]

    .rolling(7)

    .mean()

    .reset_index(level=0, drop=True)
)

df["energia_7d"] = (

    df.groupby("zona")["energia"]

    .rolling(7)

    .mean()

    .reset_index(level=0, drop=True)
)

df = df.replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

df.to_csv(
    "data/dataset_limpio.csv",
    index=False
)

print("✅ DATASET LIMPIO")
print(df.shape)