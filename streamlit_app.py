# streamlit_app.py
# -*- coding: utf-8 -*-
import pandas as pd
import plotly.express as px
import streamlit as st

# Importar lógica desacoplada (Fix 1)
from data_loader import (
    DATA_DEFAULT, 
    load_data, 
    unique_nonempty, 
    extract_unique_drivers, 
    fit_nombre
)

# ===================== Configuración de página =====================
st.set_page_config(
    page_title="F1 • Análisis de Pilotos (2024–2026)",
    layout="wide",
    page_icon="🏁",
)

# ===================== Sistema de Diseño de Color Unificado =====================
# Inspirado en Material Design & IBM Design for Data
COLOR_PRIMARY = "#1A73E8"       # Azul corporativo (Métrica principal / Foco)
COLOR_SECONDARY = "#4285F4"     # Azul secundario
COLOR_NEUTRAL = "#5F6368"       # Gris neutro para elementos secundarios/DSQ
COLOR_DANGER = "#D93025"        # Rojo para incidentes críticos (DNFs)
COLOR_WARNING = "#F9AB00"       # Amarillo/Naranja para advertencias (NCs)

# Paletas reutilizables
PALETTE_DISCRETE = ["#1A73E8", "#12B5CB", "#7C4DFF", "#FF6D00", "#46BDC6", "#E52592", "#00C853"]
PALETTE_SEQUENTIAL = "Blues"

# Layout limpio reutilizable para Plotly (Tufte / Storytelling with Data)
PLOT_LAYOUT_CLEAN = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=50, b=10),
)

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Filtros (Pilotos)")

csv_path = st.sidebar.text_input(
    "Ruta del CSV",
    value=str(DATA_DEFAULT),
    help="Ruta al archivo f1_2024_2025_2026_filled.csv.",
)

try:
    df = load_data(csv_path)
except Exception as e:
    st.error(f"❌ No se pudo cargar el CSV.\n\n**Detalle:** {e}")
    st.stop()

temp_disponibles = unique_nonempty(df["season"]) if "season" in df.columns else []
temporadas = st.sidebar.multiselect("Temporadas", options=temp_disponibles, default=temp_disponibles)

pilotos = extract_unique_drivers(df)
piloto_sel = st.sidebar.multiselect("Pilotos (pole/podio/VR)", options=pilotos, default=[])

st.sidebar.caption("Foco: comparar pilotos entre 2024 y 2026.")

# ===================== Filtrado Maestro =====================
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
st.title("F1 • 2024–2026 – Enfoque: Pilotos")
st.write(
    "Objetivo: **¿quién domina y quién mejoró de 2024 a 2026?** "
    "Métricas: victorias, poles, vueltas rápidas y conversión Pole→Victoria."
)

# ===================== KPIs =====================
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Carreras (filtro)", len(df_f))

with k2:
    if "winner_driver" in df_f.columns and df_f["winner_driver"].notna().any():
        top_winner_full = df_f["winner_driver"].dropna().value_counts().idxmax()
        st.metric("Piloto con más victorias", fit_nombre(top_winner_full, 12), help=top_winner_full)
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
        st.metric("Más vueltas rápidas", fit_nombre(top_fl_full, 12), help=top_fl_full)
    else:
        st.metric("Más vueltas rápidas", "—")

st.divider()

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

    win_drv["season"] = win_drv["season"].astype("Int64").astype(str)

    fig_win_drv = px.bar(
        win_drv.sort_values(["season", "wins"], ascending=[True, False]),
        x="winner_driver", y="wins", color="season", barmode="group",
        category_orders={"season": season_order},
        color_discrete_sequence=PALETTE_DISCRETE,  # Usando paleta unificada
        labels={"winner_driver": "Piloto", "wins": "Victorias", "season": "Temporada"},
        title="Conteo de victorias por piloto",
    )
    fig_win_drv.update_layout(
        **PLOT_LAYOUT_CLEAN,
        xaxis_tickangle=0,
        legend_title_text="Temporada"
    )
    st.plotly_chart(fig_win_drv, use_container_width=True)

st.divider()

# ===================== 2) Poles por piloto (Total) =====================
st.subheader("Poles por piloto (Total)")
if "pole_driver" not in df_f.columns or len(df_f) == 0:
    st.info("No hay columna 'pole_driver' o no hay filas tras el filtro.")
else:
    poles_total = (
        df_f.dropna(subset=["pole_driver"])
            .groupby("pole_driver", as_index=False)
            .size()
            .rename(columns={"size": "poles"})
            .sort_values("poles", ascending=True) 
    )

    if piloto_sel:
        poles_total = poles_total[poles_total["pole_driver"].isin(piloto_sel)]

    fig_poles = px.bar(
        poles_total, x="poles", y="pole_driver", orientation='h',
        title="Total de Poles por Piloto",
        color_discrete_sequence=[COLOR_PRIMARY],  # Color unificado
        text="poles",
        labels={"pole_driver": "Piloto", "poles": "Cantidad de Poles"}
    )
    fig_poles.update_traces(textposition='outside', marker_line_width=0)
    fig_poles.update_layout(
        **PLOT_LAYOUT_CLEAN,
        xaxis=dict(showgrid=True, gridcolor='rgba(200,200,200,0.2)', title="Cantidad de Poles"),
        yaxis=dict(showgrid=False, title=None),
        hovermode="y"
    )
    st.plotly_chart(fig_poles, use_container_width=True)

st.divider()

# ===================== 3) Vueltas rápidas =====================
st.subheader("Vueltas rápidas por piloto (Conteo por Circuito)")
if "fastest_lap_driver" not in df_f.columns or len(df_f) == 0:
    st.info("No hay columna 'fastest_lap_driver' o no hay filas tras el filtro.")
else:
    if 'gp_name' in df_f.columns and 'date' in df_f.columns:
        fl = (
            df_f.dropna(subset=["fastest_lap_driver", "date", "gp_name"])
                .groupby(["date", "gp_name", "fastest_lap_driver"], as_index=False)
                .size()
                .rename(columns={"size": "fastest_laps"})
        )
        if piloto_sel:
            fl = fl[fl["fastest_lap_driver"].isin(piloto_sel)]

        fl_sorted = fl.sort_values("date", ascending=True)
        gp_order = fl_sorted["gp_name"].unique().tolist()

        fig_fl = px.bar(
            fl_sorted, y="gp_name", x="fastest_laps", color="fastest_lap_driver",
            orientation='h', category_orders={"gp_name": gp_order},
            color_discrete_sequence=PALETTE_DISCRETE,  # Paleta unificada
            labels={"fastest_lap_driver": "Piloto", "fastest_laps": "Vueltas Rápidas", "gp_name": "Gran Premio"},
            title="Conteo de Vueltas Rápidas por Piloto y Gran Premio",
        )
        fig_fl.update_layout(
            **PLOT_LAYOUT_CLEAN,
            height=600,
            legend_title_text="Piloto",
            yaxis_title=None,
            xaxis_title="Vueltas Rápidas",
            yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(gp_order))}
        )
        fig_fl.update_xaxes(tick0=0, dtick=1, showgrid=True, gridcolor='rgba(200,200,200,0.2)')
        st.plotly_chart(fig_fl, use_container_width=True)

st.divider()

# ===================== 4) Conversión Pole → Victoria =====================
st.subheader("Conversión Pole → Victoria por piloto")
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

    conv["season"] = conv["season"].astype("Int64").astype(str)
    
    heatmap_data = conv.pivot_table(index='pole_driver', columns='season', values='conversion_pct', fill_value=0).round(1)
    heatmap_data['avg_conversion'] = heatmap_data.mean(axis=1)
    bar_data = heatmap_data[['avg_conversion']].reset_index().sort_values('avg_conversion', ascending=True) 
    
    st.write("### 📊 Eficiencia Promedio - Conversión Pole → Victoria")
    fig_bar = px.bar(
        bar_data, x='avg_conversion', y='pole_driver', orientation='h',
        title='Promedio de Porcentaje de Conversión Pole → Victoria por Piloto',
        color='avg_conversion',
        color_continuous_scale=PALETTE_SEQUENTIAL,  # Escala monocromática azul accesible
        labels={'avg_conversion': 'Conversión Promedio (%)', 'pole_driver': 'Piloto'},
    )
    fig_bar.update_traces(marker_line_width=0, texttemplate='%{x:.1f}%', textposition='outside')
    fig_bar.update_layout(
        **PLOT_LAYOUT_CLEAN,
        height=max(400, 35 * len(bar_data)),
        xaxis_title="Conversión Promedio (%)",
        yaxis_title=None,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)
    
    st.write("### 📜 Estadísticas Resumen Detallado")
    heatmap_data = heatmap_data.sort_values('avg_conversion', ascending=False) 
    stats_data = []
    
    for piloto in heatmap_data.index: 
        datos_piloto = heatmap_data.loc[piloto] 
        datos_totales = conv[conv['pole_driver'] == piloto] 
        datos_piloto_sin_promedio = datos_piloto.drop('avg_conversion', errors='ignore')
        datos_piloto_sin_cero = datos_piloto_sin_promedio[datos_piloto_sin_promedio > 0]
    
        stats_data.append({
            'Piloto': piloto,
            'Poles Totales': datos_totales['poles'].sum(),
            'Victorias desde Pole': datos_totales['wins_from_pole'].sum(),
            'Eficiencia Promedio': f"{datos_piloto['avg_conversion']:.1f}%",
            'Mejor Temporada': f"{datos_piloto_sin_cero.max():.1f}%" if len(datos_piloto_sin_cero) > 0 else "N/A"
        })

    st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
else:
    st.info("Faltan columnas 'pole_driver' o 'winner_driver' para este análisis.")
    
# ===================== 5) Análisis de Confiabilidad =====================
st.divider()
st.subheader("🛠️ Análisis de Confiabilidad por Temporada")
st.write("Desglose de incidentes en pista: Abandonos reales (DNFs), Descalificaciones (DSQs) y No Clasificados (NCs).")

if all(col in df_f.columns for col in ["dnfs", "dsqs", "ncs"]):
    status_df = df_f.groupby("season")[["dnfs", "dsqs", "ncs"]].sum().reset_index()
    status_melted = status_df.melt(id_vars="season", var_name="Tipo", value_name="Cantidad")
    
    status_melted["Tipo"] = status_melted["Tipo"].map({
        "dnfs": "Abandonos (DNFs)", "dsqs": "Descalificados (DSQs)", "ncs": "No Clasificados (NCs)"
    })
    
    status_melted["season"] = status_melted["season"].astype("Int64").astype(str)

    fig_status = px.bar(
        status_melted, x="season", y="Cantidad", color="Tipo", title="Total de Incidentes por Temporada",
        barmode="stack", category_orders={"season": season_order},
        # Mapa de color semántico alineado con el sistema de diseño
        color_discrete_map={
            "Abandonos (DNFs)": COLOR_DANGER,
            "Descalificados (DSQs)": COLOR_NEUTRAL,
            "No Clasificados (NCs)": COLOR_WARNING
        },
        labels={"season": "Temporada", "Cantidad": "Cantidad total", "Tipo": "Tipo de Incidente"}
    )
    fig_status.update_layout(
        **PLOT_LAYOUT_CLEAN,
        legend_title_text="Estado"
    )
    st.plotly_chart(fig_status, use_container_width=True)
else:
    st.info("⚠️ Las columnas de confiabilidad no están presentes.")
    
st.caption("Fuente: Ergast (ingesta propia). Gráficos: Plotly Express • App: Streamlit.")