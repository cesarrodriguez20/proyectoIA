# entrenamiento.py

import pandas as pd
import numpy as np
import joblib
import os

from xgboost import XGBClassifier

from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import classification_report

# ─────────────────────────────────────
# CREAR CARPETA MODELOS
# ─────────────────────────────────────

os.makedirs(
    "modelos",
    exist_ok=True
)

# ─────────────────────────────────────
# CARGAR FEATURES
# ─────────────────────────────────────

df = pd.read_csv(
    "data/features_para_ia.csv"
)

# ─────────────────────────────────────
# FEATURES IA
# ─────────────────────────────────────

features = [

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
    "energia_7d",

    "delta_energia",

    "riesgo_tectonico",

    "lag_1",
    "lag_2",
    "lag_3",

    "b_value",

    "aceleracion",

    "energia_30d"
]

# guardar nombres features

joblib.dump(
    features,
    "modelos/features_nombres.pkl"
)

# ─────────────────────────────────────
# ENTRENAMIENTO POR DEPARTAMENTO
# ─────────────────────────────────────

for zona in df["zona"].unique():

    print("\n" + "="*50)
    print(f"🌍 ENTRENANDO: {zona.upper()}")
    print("="*50)

    # filtrar zona

    df_z = (

        df[df["zona"] == zona]

        .copy()

        .sort_values("fecha")
    )

    X = df_z[features]

    y = df_z["target"]

    # ─────────────────────────────────
    # VALIDAR DATOS
    # ─────────────────────────────────

    if len(X) < 2:

        print(
            f"⚠️ {zona}: "
            "muy pocos datos"
        )

        continue

    # ─────────────────────────────────
    # VALIDAR CLASES
    # ─────────────────────────────────

    positivos = (y == 1).sum()
    negativos = (y == 0).sum()

    if positivos < 2:

        print(
            f"⚠️ {zona}: "
            "muy pocos eventos positivos"
        )

        continue

    # ─────────────────────────────────
    # SPLITS DINÁMICOS
    # ─────────────────────────────────

    splits = min(5, len(X) - 1)

    if splits < 2:

        print(
            f"⚠️ {zona}: "
            "insuficientes muestras"
        )

        continue

    tscv = TimeSeriesSplit(
        n_splits=splits
    )

    # obtener último fold

    for train_idx, test_idx in tscv.split(X):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

    # ─────────────────────────────────
    # PESO BALANCEO
    # ─────────────────────────────────

    positivos_train = (y_train == 1).sum()

    negativos_train = (y_train == 0).sum()

    scale_weight = (

        negativos_train
        /
        max(positivos_train, 1)
    )

    # ─────────────────────────────────
    # MODELO XGBOOST
    # ─────────────────────────────────

    modelo = XGBClassifier(

        n_estimators=400,

        learning_rate=0.03,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        scale_pos_weight=scale_weight,

        eval_metric="logloss",

        random_state=42
    )

    # ─────────────────────────────────
    # ENTRENAR
    # ─────────────────────────────────

    modelo.fit(

        X_train,
        y_train
    )

    # ─────────────────────────────────
    # PREDICCIÓN
    # ─────────────────────────────────

    pred = modelo.predict(
        X_test
    )

    # ─────────────────────────────────
    # REPORTE
    # ─────────────────────────────────

    print("\n📊 RESULTADOS")

    print(

        classification_report(

            y_test,
            pred,

            zero_division=0
        )
    )

    # ─────────────────────────────────
    # GUARDAR MODELO
    # ─────────────────────────────────

    ruta_modelo = (
        f"modelos/modelo_{zona}.pkl"
    )

    joblib.dump(
        modelo,
        ruta_modelo
    )

    print(
        f"✅ Modelo guardado:"
    )

    print(ruta_modelo)

# ─────────────────────────────────────
# FINAL
# ─────────────────────────────────────

print("\n🌍 ENTRENAMIENTO FINALIZADO")