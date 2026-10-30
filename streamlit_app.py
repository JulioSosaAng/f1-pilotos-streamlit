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

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ===================== Sistema de Diseño de Color =====================
COLOR_PRIMARY = "#1A73E8"       
COLOR_SECONDARY = "#4285F4"     
COLOR_NEUTRAL = "#5F6368"       
COLOR_DANGER = "#D93025"        
COLOR_WARNING = "#F9AB00"       

PALETTE_DISCRETE = ["#1A73E8", "#12B5CB", "#7C4DFF", "#FF6D00", "#46BDC6", "#E52592", "#00C853"]
PALETTE_SEQUENTIAL = "Blues"

PLOT_LAYOUT_CLEAN = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=20, t=50, b=10),
)

# ===================== Sidebar =====================
st.sidebar.header("⚙️ Filtros (Pilotos)")

csv_path = st.sidebar.text_input("Ruta del CSV", value=str(DATA_DEFAULT))

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
df_filtered = df.copy()
if temporadas:
    df_filtered = df_filtered[df_filtered["season"].isin(temporadas)]

if piloto_sel:
    cols_drv = ["winner_driver", "p2_driver", "p3_driver", "pole_driver", "fastest_lap_driver"]
    mask = pd.Series(False, index=df_filtered.index)
    for c in cols_drv:
        if c in df_filtered.columns:
            mask = mask | df_filtered[c].isin(piloto_sel)
    df_filtered = df_filtered[mask]

season_order = [str(s) for s in sorted(df_filtered["season"].dropna().unique())]

# ===================== FIX 7: Capa de Pre-cálculo y Modelado (Data Marts) =====================
# Consolidamos toda la transformación lógica aquí para no bloquear la renderización de la UI

# 1. Mart de Victorias
df_wins = pd.DataFrame()
if "winner_driver" in df_filtered.columns and not df_filtered.empty:
    df_wins = df_filtered.dropna(subset=["winner_driver"]).groupby(["season", "winner_driver"], as_index=False).size().rename(columns={"size": "wins"})
    if piloto_sel: df_wins = df_wins[df_wins["winner_driver"].isin(piloto_sel)]
    df_wins["season"] = df_wins["season"].astype("Int64").astype(str)

# 2. Mart de Poles
df_poles = pd.DataFrame()
if "pole_driver" in df_filtered.columns and not df_filtered.empty:
    df_poles = df_filtered.dropna(subset=["pole_driver"]).groupby("pole_driver", as_index=False).size().rename(columns={"size": "poles"}).sort_values("poles", ascending=True)
    if piloto_sel: df_poles = df_poles[df_poles["pole_driver"].isin(piloto_sel)]

# 3. Mart de Vueltas Rápidas
df_fastest_laps_sorted = pd.DataFrame()
gp_order = []
if all(c in df_filtered.columns for c in ["fastest_lap_driver", "date", "gp_name"]) and not df_filtered.empty:
    df_fl = df_filtered.dropna(subset=["fastest_lap_driver", "date", "gp_name"]).groupby(["date", "gp_name", "fastest_lap_driver"], as_index=False).size().rename(columns={"size": "fastest_laps"})
    if piloto_sel: df_fl = df_fl[df_fl["fastest_lap_driver"].isin(piloto_sel)]
    df_fastest_laps_sorted = df_fl.sort_values("date", ascending=True)
    gp_order = df_fastest_laps_sorted["gp_name"].unique().tolist()

# 4. Mart de Conversión (Hechos agregados complejos)
bar_data = pd.DataFrame()
stats_data = []
if {"pole_driver", "winner_driver"}.issubset(df_filtered.columns) and not df_filtered.empty:
    df_tmp = df_filtered.copy()
    df_tmp["pole_won"] = (df_tmp["pole_driver"] == df_tmp["winner_driver"]).astype(int)
    df_conversion = df_tmp.dropna(subset=["pole_driver"]).groupby(["season", "pole_driver"], as_index=False).agg(poles=("pole_driver", "size"), wins_from_pole=("pole_won", "sum"))
    df_conversion["conversion_pct"] = (df_conversion["wins_from_pole"] / df_conversion["poles"]).fillna(0.0) * 100.0
    if piloto_sel: df_conversion = df_conversion[df_conversion["pole_driver"].isin(piloto_sel)]
    
    if not df_conversion.empty:
        heatmap_data = df_conversion.pivot_table(index='pole_driver', columns='season', values='conversion_pct', fill_value=0).round(1)
        heatmap_data['avg_conversion'] = heatmap_data.mean(axis=1)
        bar_data = heatmap_data[['avg_conversion']].reset_index().sort_values('avg_conversion', ascending=True) 
        
        heatmap_data_sorted = heatmap_data.sort_values('avg_conversion', ascending=False)
        for piloto in heatmap_data_sorted.index:
            datos_piloto = heatmap_data_sorted.loc[piloto] 
            datos_totales = df_conversion[df_conversion['pole_driver'] == piloto] 
            datos_piloto_sin_promedio = datos_piloto.drop('avg_conversion', errors='ignore')
            datos_piloto_sin_cero = datos_piloto_sin_promedio[datos_piloto_sin_promedio > 0]
            stats_data.append({
                'Piloto': piloto, 'Poles Totales': datos_totales['poles'].sum(),
                'Victorias desde Pole': datos_totales['wins_from_pole'].sum(),
                'Eficiencia Promedio': f"{datos_piloto['avg_conversion']:.1f}%",
                'Mejor Temporada': f"{datos_piloto_sin_cero.max():.1f}%" if not datos_piloto_sin_cero.empty else "N/A"
            })

# 5. Mart de Confiabilidad
status_melted = pd.DataFrame()
if all(col in df_filtered.columns for col in ["dnfs", "dsqs", "ncs"]) and not df_filtered.empty:
    status_df = df_filtered.groupby("season")[["dnfs", "dsqs", "ncs"]].sum().reset_index()
    status_melted = status_df.melt(id_vars="season", var_name="Tipo", value_name="Cantidad")
    status_melted["Tipo"] = status_melted["Tipo"].map({"dnfs": "Abandonos (DNFs)", "dsqs": "Descalificados (DSQs)", "ncs": "No Clasificados (NCs)"})
    status_melted["season"] = status_melted["season"].astype("Int64").astype(str)

# ===================== Encabezado =====================
st.title("F1 • 2024–2026 – Enfoque: Pilotos")
st.write("Objetivo: **¿quién domina y quién mejoró de 2024 a 2026?**")

# ===================== KPIs =====================
col_metric_1, col_metric_2, col_metric_3, col_metric_4 = st.columns(4)

with col_metric_1: st.metric("Carreras (filtro)", len(df_filtered))
with col_metric_2:
    if not df_wins.empty: st.metric("Piloto con más victorias", fit_nombre(df_wins.groupby('winner_driver')['wins'].sum().idxmax(), 12))
    else: st.metric("Piloto con más victorias", "—")
with col_metric_3:
    if not bar_data.empty and len(df_filtered): st.metric("Pole → Victoria (global)", f"{(df_filtered['pole_driver'] == df_filtered['winner_driver']).mean() * 100.0:.1f}%")
    else: st.metric("Pole → Victoria (global)", "—")
with col_metric_4:
    if not df_fastest_laps_sorted.empty: st.metric("Más vueltas rápidas", fit_nombre(df_fastest_laps_sorted.groupby('fastest_lap_driver')['fastest_laps'].sum().idxmax(), 12))
    else: st.metric("Más vueltas rápidas", "—")

st.divider()

# ===================== Estructura de Pestañas (UI Limpia) =====================
tab_victorias, tab_rendimiento, tab_confiabilidad = st.tabs(["🏆 Victorias y Poles", "⏱️ Rendimiento y Conversión", "🛠️ Confiabilidad"])

with tab_victorias:
    st.subheader("Victorias por piloto (agrupado por temporada)")
    if df_wins.empty: st.info("No hay datos de victorias tras el filtro.")
    else:
        fig_win_drv = px.bar(
            df_wins.sort_values(["season", "wins"], ascending=[True, False]), x="winner_driver", y="wins", color="season", barmode="group",
            category_orders={"season": season_order}, color_discrete_sequence=PALETTE_DISCRETE, title="Conteo de victorias por piloto"
        )
        fig_win_drv.update_traces(hovertemplate="<b>%{x}</b><br>Temporada: %{data.name}<br>Victorias: <b>%{y}</b><extra></extra>", marker_line_width=0)
        fig_win_drv.update_layout(**PLOT_LAYOUT_CLEAN, xaxis=dict(showgrid=False, zeroline=False, title=None), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None))
        st.plotly_chart(fig_win_drv, use_container_width=True)

    st.divider()
    st.subheader("Poles por piloto (Total)")
    if df_poles.empty: st.info("No hay datos de poles tras el filtro.")
    else:
        fig_poles = px.bar(df_poles, x="poles", y="pole_driver", orientation='h', title="Total de Poles por Piloto", color_discrete_sequence=[COLOR_PRIMARY])
        fig_poles.update_traces(hovertemplate="<b>%{y}</b><br>Poles: <b>%{x}</b><extra></extra>", marker_line_width=0)
        fig_poles.update_layout(**PLOT_LAYOUT_CLEAN, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None), yaxis=dict(showgrid=False, zeroline=False, title=None))
        st.plotly_chart(fig_poles, use_container_width=True)

with tab_rendimiento:
    st.subheader("Vueltas rápidas por piloto (Conteo por Circuito)")
    if df_fastest_laps_sorted.empty: st.info("No hay datos de vueltas rápidas tras el filtro.")
    else:
        fig_fl = px.bar(
            df_fastest_laps_sorted, y="gp_name", x="fastest_laps", color="fastest_lap_driver", orientation='h', category_orders={"gp_name": gp_order},
            color_discrete_sequence=PALETTE_DISCRETE, title="Conteo de Vueltas Rápidas por Piloto y Gran Premio"
        )
        fig_fl.update_traces(marker_line_width=0, hovertemplate="<b>%{y}</b><br>Piloto: %{data.name}<br>Vueltas: <b>%{x}</b><extra></extra>")
        fig_fl.update_layout(**PLOT_LAYOUT_CLEAN, height=600, legend_title_text="Piloto", yaxis={'categoryorder': 'array', 'categoryarray': list(reversed(gp_order))}, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None), yaxis_title=None)
        st.plotly_chart(fig_fl, use_container_width=True)

    st.divider()
    st.subheader("Conversión Pole → Victoria por piloto")
    if bar_data.empty: st.info("No hay datos de conversión tras el filtro.")
    else:
        st.write("### 📊 Eficiencia Promedio")
        fig_bar = px.bar(
            bar_data, x='avg_conversion', y='pole_driver', orientation='h', title='Promedio de Porcentaje de Conversión Pole → Victoria por Piloto',
            color='avg_conversion', color_continuous_scale=PALETTE_SEQUENTIAL
        )
        fig_bar.update_traces(marker_line_width=0, texttemplate='%{x:.1f}%', textposition='outside')
        fig_bar.update_layout(**PLOT_LAYOUT_CLEAN, height=max(400, 35 * len(bar_data)), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None), yaxis=dict(showgrid=False, zeroline=False, title=None), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write("### 📜 Estadísticas Resumen Detallado")
        st.dataframe(pd.DataFrame(stats_data), use_container_width=True)

with tab_confiabilidad:
    st.subheader("🛠️ Análisis de Confiabilidad por Temporada")
    if status_melted.empty: st.info("⚠️ Las columnas de confiabilidad no están presentes o no hay datos.")
    else:
        fig_status = px.bar(
            status_melted, x="season", y="Cantidad", color="Tipo", title="Total de Incidentes por Temporada", barmode="stack", category_orders={"season": season_order},
            color_discrete_map={"Abandonos (DNFs)": COLOR_DANGER, "Descalificados (DSQs)": COLOR_NEUTRAL, "No Clasificados (NCs)": COLOR_WARNING}
        )
        fig_status.update_traces(marker_line_width=0, hovertemplate="<b>%{x}</b><br>Estado: %{data.name}<br>Cantidad: <b>%{y}</b><extra></extra>")
        fig_status.update_layout(**PLOT_LAYOUT_CLEAN, legend_title_text="Estado", yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=None), xaxis=dict(showgrid=False, zeroline=False, title=None))
        st.plotly_chart(fig_status, use_container_width=True)