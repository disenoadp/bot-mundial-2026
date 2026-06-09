import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# ==========================================
# 1. LÓGICA MATEMÁTICA (Modelo Estadístico)
# ==========================================
# Valores estimados de ataque y defensa para las selecciones
ESTADISTICAS_SELECCIONES = {
    "Argentina": {"ofensiva": 2.2, "defensiva": 0.5},
    "Francia": {"ofensiva": 2.1, "defensiva": 0.6},
    "España": {"ofensiva": 2.0, "defensiva": 0.7},
    "México": {"ofensiva": 1.4, "defensiva": 1.0},
    "Estados Unidos": {"ofensiva": 1.5, "defensiva": 0.9},
    "Marruecos": {"ofensiva": 1.3, "defensiva": 0.8},
    "Japón": {"ofensiva": 1.6, "defensiva": 0.9},
    "Sudáfrica": {"ofensiva": 1.0, "defensiva": 1.3},
    "Brasil": {"ofensiva": 2.1, "defensiva": 0.7},
    "Alemania": {"ofensiva": 1.9, "defensiva": 0.9},
}

def predecir_partido(local, visitante):
    default = {"ofensiva": 1.2, "defensiva": 1.2}
    stats_local = ESTADISTICAS_SELECCIONES.get(local, default)
    stats_vis = ESTADISTICAS_SELECCIONES.get(visitante, default)
    
    goles_esperados_local = stats_local["ofensiva"] * stats_vis["defensiva"]
    goles_esperados_visitante = stats_vis["ofensiva"] * stats_local["defensiva"]
    
    prob_local, prob_empate, prob_visitante = 0, 0, 0
    
    for g_local in range(8):
        for g_vis in range(8):
            p_l = poisson.pmf(g_local, goles_esperados_local)
            p_v = poisson.pmf(g_vis, goles_esperados_visitante)
            prob_marcador = p_l * p_v
            
            if g_local > g_vis:
                prob_local += prob_marcador
            elif g_local < g_vis:
                prob_visitante += prob_marcador
            else:
                prob_empate += prob_marcador
                
    return {
        "Local": round(prob_local * 100, 1),
        "Empate": round(prob_empate * 100, 1),
        "Visitante": round(prob_visitante * 100, 1)
    }

# ==========================================
# 2. DISEÑO DEL DASHBOARD INTERACTIVO
# ==========================================
st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")

st.title("⚽ Bot Predictor Online - Mundial 2026")
st.write("Predicciones instantáneas para la Fase de Grupos generadas por Inteligencia Estadística.")
st.markdown("---")

# Partidos fijos iniciales de ejemplo
PARTIDOS_FASE_DE_GRUPOS = [
    {"Grupo": "Grupo A", "Local": "México", "Visitante": "Sudáfrica"},
    {"Grupo": "Grupo A", "Local": "Francia", "Visitante": "Japón"},
    {"Grupo": "Grupo B", "Local": "Estados Unidos", "Visitante": "Marruecos"},
    {"Grupo": "Grupo B", "Local": "Argentina", "Visitante": "España"},
    {"Grupo": "Grupo C", "Local": "Brasil", "Visitante": "Alemania"},
    {"Grupo": "Grupo C", "Local": "México", "Visitante": "Francia"},
]

df_partidos = pd.DataFrame(PARTIDOS_FASE_DE_GRUPOS)

# Filtro en barra lateral
st.sidebar.header("Filtros del Torneo")
grupos_disponibles = ["Todos"] + list(df_partidos["Grupo"].unique())
grupo_sel = st.sidebar.selectbox("Selecciona un Grupo:", grupos_disponibles)

df_filtrado = df_partidos if grupo_sel == "Todos" else df_partidos[df_partidos["Grupo"] == grupo_sel]

st.subheader("📊 Partidos Programados e Interactivos")
st.write("Haz clic en cualquier partido para desplegar el análisis probabilístico:")

for index, fila in df_filtrado.iterrows():
    local = fila["Local"]
    visitante = fila["Visitante"]
    grupo = fila["Grupo"]
    
    pred = predecir_partido(local, visitante)
    
    with st.expander(f"🔍 {grupo}: {local} vs {visitante}"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label=f"Probabilidad {local}", value=f"{pred['Local']}%")
            st.progress(int(pred['Local']))
        with col2:
            st.metric(label="Probabilidad Empate", value=f"{pred['Empate']}%")
            st.progress(int(pred['Empate']))
        with col3:
            st.metric(label=f"Probabilidad {visitante}", value=f"{pred['Visitante']}%")
            st.progress(int(pred['Visitante']))
