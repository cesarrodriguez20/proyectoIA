# dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium

from folium.plugins import HeatMap
from streamlit_folium import st_folium
from streamlit_autorefresh import st_autorefresh

from datetime import datetime

# ─────────────────────────────────────
# CONFIG
# ─────────────────────────────────────

st.set_page_config(
    page_title="TECTONIC IA",
    layout="wide"
)

st_autorefresh(
    interval=60000,
    key="refresh"
)

# ─────────────────────────────────────
# ESTILO
# ─────────────────────────────────────

st.markdown("""

<style>

.main {
    background-color: #0b1220;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

div[data-testid="stMetric"] {

    background-color: #1f2937;

    border: 1px solid #374151;

    border-radius: 12px;

    padding: 15px;
}

h1,h2,h3,h4 {
    color: white;
}

</style>

""", unsafe_allow_html=True)

# ─────────────────────────────────────
# DEPARTAMENTOS
# ─────────────────────────────────────

departamentos = {

    "santander": {
        "nombre": "Santander",
        "lat": 6.833,
        "lon": -73.069
    },

    "tolima": {
        "nombre": "Tolima",
        "lat": 4.438,
        "lon": -75.232
    },

    "risaralda": {
        "nombre": "Risaralda",
        "lat": 4.814,
        "lon": -75.694
    },

    "narino": {
        "nombre": "Nariño",
        "lat": 1.214,
        "lon": -77.278
    },

    "choco": {
        "nombre": "Chocó",
        "lat": 5.694,
        "lon": -76.661
    },

    "antioquia": {
        "nombre": "Antioquia",
        "lat": 6.251,
        "lon": -75.563
    },

    "cauca": {
        "nombre": "Cauca",
        "lat": 2.445,
        "lon": -76.614
    }
}

# ─────────────────────────────────────
# CARGAR DATASET
# ─────────────────────────────────────

@st.cache_data
def cargar_datos():

    df = pd.read_csv(
        "data/dataset_limpio.csv"
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df

# ─────────────────────────────────────
# HOTSPOTS
# ─────────────────────────────────────

@st.cache_data
def cargar_hotspots():

    try:

        return pd.read_csv(
            "data/hotspots.csv"
        )

    except:

        return pd.DataFrame()

# ─────────────────────────────────────
# MODELOS
# ─────────────────────────────────────

@st.cache_resource
def cargar_modelo(zona):

    return joblib.load(
        f"modelos/modelo_{zona}.pkl"
    )

# ─────────────────────────────────────
# FEATURES
# ─────────────────────────────────────

features = joblib.load(
    "modelos/features_nombres.pkl"
)

df = cargar_datos()

hotspots = cargar_hotspots()

# ─────────────────────────────────────
# CONSTRUIR ENTRADA IA
# ─────────────────────────────────────

def construir_entrada(df_z):

    ult = df_z.tail(24)

    energia_total = ult["energia"].sum()

    return pd.DataFrame([{

        "conteo":
            len(ult),

        "mag_promedio":
            ult["magnitude"].mean(),

        "mag_max":
            ult["magnitude"].max(),

        "depth_promedio":
            ult["depth"].mean(),

        "depth_std":
            ult["depth"].std(),

        "depth_max":
            ult["depth"].max(),

        "energia_total":
            energia_total,

        "energia_max":
            ult["energia"].max(),

        "delta_t":
            ult["delta_t"].mean(),

        "mag_7d":
            ult["mag_7d"].mean(),

        "energia_7d":
            ult["energia_7d"].mean(),

        "delta_energia":
            ult["energia"].iloc[-1]
            -
            ult["energia"].iloc[0],

        "riesgo_tectonico":
            energia_total * 1.5,

        "lag_1":
            len(ult),

        "lag_2":
            len(ult),

        "lag_3":
            len(ult),

        "b_value":
            np.log10(np.e)
            /
            (
                ult["magnitude"].mean()
                -
                ult["magnitude"].min()
                +
                0.001
            ),

        "aceleracion":
            ult["magnitude"].diff().mean(),

        "energia_30d":
            df_z.tail(30)["energia"].sum()

    }])

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────

st.sidebar.title(
    "TECTONIC IA"
)

zona = st.sidebar.selectbox(

    "Departamento",

    sorted(df["zona"].unique()),

    format_func=lambda x:
        departamentos[x]["nombre"]
)

st.sidebar.markdown("---")

st.sidebar.caption(

    f"Actualización: "
    f"{datetime.now().strftime('%H:%M:%S')}"
)

# ─────────────────────────────────────
# RESUMEN NACIONAL
# ─────────────────────────────────────

st.subheader(
    "🇨🇴 Riesgo Sísmico Nacional"
)

cols = st.columns(7)

for i, zona_name in enumerate(
    departamentos.keys()
):

    try:

        modelo = cargar_modelo(
            zona_name
        )

        df_z = (

            df[df["zona"] == zona_name]

            .sort_values("date")
        )

        entrada = construir_entrada(
            df_z
        )

        p = modelo.predict_proba(
            entrada
        )[0][1]

        icono = (

            "🟢"
            if p < 0.30
            else "🟡"
            if p < 0.70
            else "🔴"
        )

        cols[i].metric(

            f"{icono} {departamentos[zona_name]['nombre']}",

            f"{p*100:.0f}%"
        )

    except:

        cols[i].metric(
            zona_name,
            "—"
        )

st.divider()

# ─────────────────────────────────────
# ZONA SELECCIONADA
# ─────────────────────────────────────

df_z = (

    df[df["zona"] == zona]

    .sort_values("date")
)

modelo = cargar_modelo(
    zona
)

entrada = construir_entrada(
    df_z
)

prob = modelo.predict_proba(
    entrada
)[0][1]

dep = departamentos[zona]

# ─────────────────────────────────────
# TÍTULO
# ─────────────────────────────────────

st.title(

    f"🌍 {dep['nombre'].upper()}"
)

# ─────────────────────────────────────
# ALERTA IA
# ─────────────────────────────────────

hotspot_zona = hotspots[
    hotspots["zona"] == zona
]

if len(hotspot_zona) > 0:

    h = hotspot_zona.iloc[0]

    if prob < 0.30:

        st.success(

            f"""
🟢 ACTIVIDAD BAJA

Probabilidad próximas 24h:
{prob*100:.1f}%
"""
        )

    elif prob < 0.70:

        st.warning(

            f"""
🟡 ACTIVIDAD MODERADA

Probabilidad próximas 24h:
{prob*100:.1f}%
"""
        )

    else:

        st.error(

            f"""
🔴 ALERTA SÍSMICA

Departamento:
{dep['nombre']}

📍 Zona crítica detectada

Latitud:
{h['latitud_hotspot']:.4f}

Longitud:
{h['longitud_hotspot']:.4f}

🌋 Eventos agrupados:
{int(h['eventos_cluster'])}

⚡ Energía acumulada:
{h['energia_cluster']:.2e}

🧭 Migración:
{h['direccion_migracion']}

⏳ Probabilidad próximas 24h:
{prob*100:.1f}%
"""
        )

# ─────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Eventos",
    len(df_z)
)

c2.metric(
    "Magnitud Máxima",
    f"{df_z['magnitude'].max():.2f}"
)

c3.metric(
    "Profundidad Prom.",
    f"{df_z['depth'].mean():.2f} km"
)

c4.metric(
    "Energía Total",
    f"{df_z['energia'].sum():.2e}"
)

c5.metric(
    "Predicción IA",
    f"{prob*100:.1f}%"
)

st.divider()

# ─────────────────────────────────────
# GRÁFICAS
# ─────────────────────────────────────

g1, g2 = st.columns(2)

with g1:

    st.subheader(
        "📈 Magnitud Promedio"
    )

    st.line_chart(

        df_z.groupby(

            df_z["date"].dt.date

        )["magnitude"].mean()
    )

with g2:

    st.subheader(
        "⚡ Energía Acumulada"
    )

    st.area_chart(

        df_z.groupby(

            df_z["date"].dt.date

        )["energia"].sum()
    )

st.divider()

# ─────────────────────────────────────
# MAPA
# ─────────────────────────────────────

st.subheader(
    "🗺️ Mapa Sísmico Inteligente"
)

m = folium.Map(

    location=[
        dep["lat"],
        dep["lon"]
    ],

    zoom_start=8,

    tiles="CartoDB dark_matter"
)

# EVENTOS

heat_data = [

    [

        r["latitude"],
        r["longitude"],
        r["energia"] / 1e10

    ]

    for _, r in df_z.iterrows()
]

HeatMap(

    heat_data,

    radius=18,

    blur=12,

    min_opacity=0.4

).add_to(m)

# HOTSPOT

if len(hotspot_zona) > 0:

    folium.CircleMarker(

        location=[

            h["latitud_hotspot"],
            h["longitud_hotspot"]

        ],

        radius=18,

        color="red",

        fill=True,

        fill_opacity=0.9,

        popup=f"""

HOTSPOT SÍSMICO

Zona:
{zona}

Eventos:
{int(h['eventos_cluster'])}

Migración:
{h['direccion_migracion']}

"""

    ).add_to(m)

    # radio hotspot

    folium.Circle(

        location=[

            h["latitud_hotspot"],
            h["longitud_hotspot"]

        ],

        radius=80000,

        color="red",

        fill=True,

        fill_opacity=0.08

    ).add_to(m)

# EVENTO MAYOR

evento = df_z.loc[
    df_z["energia"].idxmax()
]

folium.Circle(

    location=[

        evento["latitude"],
        evento["longitude"]

    ],

    radius=50000,

    color="orange",

    fill=True,

    fill_opacity=0.2,

    popup=f"""

Magnitud:
{evento['magnitude']}

"""
).add_to(m)

# MOSTRAR

st_folium(

    m,

    width=1400,

    height=700
)

st.divider()

# ─────────────────────────────────────
# TABLA
# ─────────────────────────────────────

st.subheader(
    "📋 Eventos Recientes"
)

st.dataframe(

    df_z[[
        "date",
        "magnitude",
        "depth",
        "energia"
    ]]

    .sort_values(
        "date",
        ascending=False
    )

    .head(20),

    use_container_width=True
)

# ─────────────────────────────────────
# FOOTER
# ─────────────────────────────────────

st.caption(
    "TECTONIC  IA "
)