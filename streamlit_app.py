# streamlit_app.py
# -*- coding: utf-8 -*-
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import re

# ===================== Configuración de página =====================
st.set_page_config(
    page_title="F1 • 2024–2025 (hasta Monza) – Pilotos",
    layout="wide",
    page_icon="🏁",
)

# ===================== Utilidades / Carga de datos =====================
PROJECT_ROOT = Path(__file__).parent
DATA_DEFAULT = PROJECT_ROOT / "Data" / "f1_2024_2025_filled.csv"

@st.cache_data(show_spinner=True)
def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo: {csv_path}")

    df = pd.read_csv(csv_path)

    # Tipos y limpieza mínima
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # A numérico lo que corresponde
    for c in ("season", "round", "dnfs"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "avg_pitstops_per_driver" in df.columns:
        df["avg_pitstops_per_driver"] = pd.to_numeric(df["avg_pitstops_per_driver"], errors="coerce")

    # Etiqueta útil para ejes
    if {"gp_name", "round"}.issubset(df.columns):
        df["gp_label"] = df["round"].fillna(0).astype("Int64").astype(str).str.zfill(2) + " • " + df["gp_name"].fillna("")

    # Sprint a booleano (si existiera)
    if "sprint_weekend" in df.columns:
        df["sprint_bool"] = df["sprint_weekend"].astype(str).str.lower().isin(["si", "sí", "yes", "true", "1"])

    # Orden estable
    if "season" in df.columns and "round" in df.columns:
        df = df.sort_values(["season", "round"], na_position="last").reset_index(drop=True)

    return df

def unique_nonempty(series: pd.Series):
    return sorted(x for x in series.dropna().unique().tolist() if str(x).strip() != "")

def extract_unique_drivers(df: pd.DataFrame):
    cols = ["winner_driver", "p2_driver", "p3_driver", "pole_driver", "fastest_lap_driver"]
    s = pd.Series(dtype="string")
    for c in cols:
        if c in df.columns:
            s = pd.concat([s, df[c].astype(str)], ignore_index=True)
    return unique_nonempty(s)

def _apellido(name: str) -> str:
    if not isinstance(name, str):
        return "—"
    name = name.strip()
    if not name:
        return "—"
    drop = {"jr", "jr.", "sr", "sr.", "iii", "ii"}
    tokens = re.split(r"\s+", name)
    if tokens and tokens[-1].lower() in drop and len(tokens) > 1:
        tokens = tokens[:-1]
    return tokens[-1]

def fit_nombre(name: str, max_chars: int = 12) -> str:  # baja el umbral
    if not isinstance(name, str) or not name.strip():
        return "—"
    return name if len(name) <= max_chars else _apellido(name)

# ===================== Sidebar (foco Pilotos) =====================
st.sidebar.header("⚙️ Filtros (Pilotos)")

csv_path = st.sidebar.text_input(
    "Ruta del CSV",
    value=str(DATA_DEFAULT),
    help="Ruta al archivo f1_2024_2025_filled.csv (podés dejar la predeterminada).",
)

# Cargar
try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"❌ No se pudo cargar el CSV.\n\n**Detalle:** {e}")
    st.stop()

# Filtros visibles: SOLO temporadas y pilotos
temp_disponibles = unique_nonempty(df["season"]) if "season" in df.columns else []
temporadas = st.sidebar.multiselect("Temporadas", options=temp_disponibles, default=temp_disponibles)

pilotos = extract_unique_drivers(df)
piloto_sel = st.sidebar.multiselect("Pilotos (pole/podio/VR)", options=pilotos, default=[])

st.sidebar.caption("Foco: comparar pilotos entre 2024 y 2025 (hasta Monza).")

# ===================== Filtrado =====================
df_f = df.copy()
if temporadas:
    df_f = df_f[df_f["season"].isin(temporadas)]

if piloto_sel:
    cols_drv = ["winner_driver", "p2_driver", "p3_driver", "pole_driver", "fastest_lap_driver"]
    mask = pd.Series(False, index=df_f.index)
    for c in cols_drv:
        if c in df_f.columns:
            mask = mask | df_f[c].isin(piloto_sel)
    df_f = df_f[mask]

season_order = [str(s) for s in sorted(df_f["season"].dropna().unique())]

# ===================== Encabezado =====================
st.title("F1 • 2024–2025 (hasta Monza) – Enfoque: Pilotos")
st.write(
    "Objetivo: **¿quién domina y quién mejoró de 2024 a 2025 (hasta Monza)?** "
    "Métricas: victorias, poles, vueltas rápidas y conversión Pole→Victoria."
)

# ===================== KPIs =====================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Carreras (filtro)", len(df_f))

with k2:
    if "winner_driver" in df_f.columns and df_f["winner_driver"].notna().any():
        top_winner_full = df_f["winner_driver"].dropna().value_counts().idxmax()
        st.metric("Piloto con más victorias",fit_nombre(top_winner_full, 12),help=top_winner_full)
    else:
        st.metric("Piloto con más victorias", "—")

with k3:
    if {"pole_driver", "winner_driver"}.issubset(df_f.columns) and len(df_f):
        conv_global = (df_f["pole_driver"] == df_f["winner_driver"]).mean() * 100.0
        st.metric("Pole → Victoria (global)", f"{conv_global:.1f}%")
    else:
        st.metric("Pole → Victoria (global)", "—")

with k4:
    if "fastest_lap_driver" in df_f.columns and df_f["fastest_lap_driver"].notna().any():
        top_fl_full = df_f["fastest_lap_driver"].dropna().value_counts().idxmax()
        st.metric("Más vueltas rápidas",fit_nombre(top_fl_full, 12),help=top_fl_full)
    else:
        st.metric("Más vueltas rápidas", "—")


# ===================== 1) Victorias por piloto =====================
st.subheader("Victorias por piloto (agrupado por temporada)")
if "winner_driver" not in df_f.columns or len(df_f) == 0:
    st.info("Faltan datos de 'winner_driver' o no hay filas tras el filtro.")
else:
    win_drv = (
        df_f.dropna(subset=["winner_driver"])
           .groupby(["season", "winner_driver"], as_index=False)
           .size()
           .rename(columns={"size": "wins"})
    )
    if piloto_sel:
        win_drv = win_drv[win_drv["winner_driver"].isin(piloto_sel)]

    win_drv["season"] = win_drv["season"].astype("Int64").astype(str)  # <- NUEVO

    fig_win_drv = px.bar(
        win_drv.sort_values(["season", "wins"], ascending=[True, False]),
        x="winner_driver", y="wins", color="season", barmode="group",
        category_orders={"season": season_order},                      # <- NUEVO
        color_discrete_sequence=px.colors.qualitative.Set2,            # <- NUEVO (paleta discreta)
        labels={"winner_driver": "Piloto", "wins": "Victorias", "season": "Temporada"},
        title="Conteo de victorias por piloto",
        )
    fig_win_drv.update_layout(margin=dict(l=10, r=10, t=60, b=10), xaxis_tickangle=45, legend_title_text="Temporada")
    st.plotly_chart(fig_win_drv, use_container_width=True)

st.divider()

# ===================== 2) Poles y Vueltas Rápidas =====================
st.subheader("Poles por piloto (Total)")
if "pole_driver" not in df_f.columns or len(df_f) == 0:
    st.info("No hay columna 'pole_driver' o no hay filas tras el filtro.")
else:
    # Agrupar el total de poles por piloto (ignorando la temporada para el gráfico de torta)
    poles_total = (
        df_f.dropna(subset=["pole_driver"])
            .groupby("pole_driver", as_index=False)
            .size()
            .rename(columns={"size": "poles"})
            .sort_values("poles", ascending=False)
    )

    if piloto_sel:
        poles_total = poles_total[poles_total["pole_driver"].isin(piloto_sel)]

    # === Gráfico de Torta (Pie Chart) para Poles ===
    fig_poles = px.pie(
        poles_total,
        names="pole_driver",
        values="poles",
        color="pole_driver",
        color_discrete_sequence=px.colors.qualitative.Pastel, # Secuencia de colores más apropiada para torta
        labels={"pole_driver": "Piloto", "poles": "Poles"},
        title="Total de Poles por Piloto",
    )
    
    fig_poles.update_traces(textposition='inside', textinfo='percent+label') # Muestra porcentaje y etiqueta
    
    fig_poles.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        legend_title_text="Piloto"
    )
    st.plotly_chart(fig_poles, use_container_width=True)

st.divider()

st.subheader("Vueltas rápidas por piloto (Tendencia por Temporada)")
if "fastest_lap_driver" not in df_f.columns or len(df_f) == 0:
    st.info("No hay columna 'fastest_lap_driver' o no hay filas tras el filtro.")
else:
    # Agrupar por temporada y piloto (necesario para el gráfico de líneas)
    fl = (
        df_f.dropna(subset=["fastest_lap_driver"])
            .groupby(["season", "fastest_lap_driver"], as_index=False)
            .size()
            .rename(columns={"size": "fastest_laps"})
    )
    if piloto_sel:
        fl = fl[fl["fastest_lap_driver"].isin(piloto_sel)]

    fl["season"] = fl["season"].astype("Int64").astype(str)

    # === Gráfico de Líneas (Line Chart) para Vueltas Rápidas ===
    fig_fl = px.line(
        fl.sort_values("season", ascending=True),
        x="season", 
        y="fastest_laps", 
        color="fastest_lap_driver", # Color para diferenciar la línea de cada piloto
        markers=True, # Mostrar marcadores en los puntos de datos
        category_orders={"season": season_order},
        color_discrete_sequence=px.colors.qualitative.Dark24, # Secuencia de colores
        labels={
            "fastest_lap_driver": "Piloto",
            "fastest_laps": "VR",
            "season": "Temporada"
        },
        title="Vueltas Rápidas por Piloto a lo largo de las Temporadas",
    )
    
    fig_fl.update_layout(
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_tickangle=45,
        legend_title_text="Piloto",
        xaxis_title="Temporada",
        yaxis_title="Vueltas Rápidas (VR)"
    )
    st.plotly_chart(fig_fl, use_container_width=True)

st.divider()

# ===================== 3) Conversión Pole → Victoria =====================
st.subheader("Conversión Pole → Victoria por piloto (solo pilotos con al menos 1 pole)")
if {"pole_driver", "winner_driver"}.issubset(df_f.columns) and len(df_f):
    df_tmp = df_f.copy()
    df_tmp["pole_won"] = (df_tmp["pole_driver"] == df_tmp["winner_driver"]).astype(int)

    conv = (
        df_tmp.dropna(subset=["pole_driver"])
             .groupby(["season", "pole_driver"], as_index=False)
             .agg(poles=("pole_driver", "size"), wins_from_pole=("pole_won", "sum"))
    )
    conv["conversion_pct"] = (conv["wins_from_pole"] / conv["poles"]).fillna(0.0) * 100.0

    if piloto_sel:
        conv = conv[conv["pole_driver"].isin(piloto_sel)]

    conv["season"] = conv["season"].astype("Int64").astype(str)        # <- NUEVO

    fig_conv = px.bar(
        conv.sort_values(["season", "conversion_pct"], ascending=[True, False]),
        x="pole_driver", y="conversion_pct", color="season", barmode="group",
        category_orders={"season": season_order},                      # <- NUEVO
        color_discrete_sequence=px.colors.qualitative.Set2,            # <- NUEVO
        labels={"pole_driver": "Piloto", "conversion_pct": "Conversión (%)", "season": "Temporada"},
        title="Porcentaje de poles que terminan en victoria",
    )
    fig_conv.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, 100])    

    st.plotly_chart(fig_conv, use_container_width=True)
else:
    st.info("Faltan columnas 'pole_driver' o 'winner_driver' para este análisis.")

# ===================== Pie =====================
st.caption("Fuente: Ergast (ingesta propia). Gráficos: Plotly Express • App: Streamlit.")


