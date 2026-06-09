import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

# Configuración del Dashboard
st.set_page_config(page_title="Predictor Mundial 2026", page_icon="⚽", layout="wide")

# ==========================================
# 1. CARGA DE DATOS ESTADÍSTICOS REALES
# ==========================================
@st.cache_data
def cargar_datos():
    # Lee el archivo CSV que creamos en GitHub
    try:
        return pd.read_csv("datos_equipos.csv", index_col="Seleccion")
    except:
        # Datos de respaldo por si el archivo no se lee correctamente
        st.error("No se pudo cargar el archivo CSV. Usando datos internos.")
        return pd.DataFrame()

datos_selecciones = cargar_datos()

# ==========================================
# 2. LÓGICA DE PREDICCIÓN NUMÉRICA EXACTA
# ==========================================
def predecir_marcador_y_probabilidades(local, visitante):
    # Valores neutros por defecto si el equipo no está en el CSV
    default = {"Goles_Favor_Promedio": 1.2, "Goles_Contra_Promedio": 1.2, "Puntos_Elo": 1600}
    
    stats_l = datos_selecciones.loc[local] if local in datos_selecciones.index else pd.Series(default)
    stats_v = datos_selecciones.loc[visitante] if visitante in datos_selecciones.index else pd.Series(default)
    
    # El Factor Elo ajusta la fuerza de ataque en base al peso histórico de la selección
    factor_elo_l = stats_l["Puntos_Elo"] / stats_v["Puntos_Elo"]
    factor_elo_v = stats_v["Puntos_Elo"] / stats_l["Puntos_Elo"]
    
    # Goles esperados ajustados estadísticamente
    goles_esperados_l = stats_l["Goles_Favor_Promedio"] * stats_v["Goles_Contra_Promedio"] * factor_elo_l
    goles_esperados_v = stats_v["Goles_Favor_Promedio"] * stats_l["Goles_Contra_Promedio"] * factor_elo_v
    
    prob_local, prob_empate, prob_visitante = 0, 0, 0
    max_prob_marcador = 0
    marcador_exacto = (0, 0)
    
    # Buscamos el marcador numérico exacto con mayor probabilidad matemática
    for g_local in range(6):
        for g_vis in range(6):
            p_l = poisson.pmf(g_local, goles_esperados_l)
            p_v = poisson.pmf(g_vis, goles_esperados_v)
            prob_marcador = p_l * p_v
            
            # Evaluar el marcador exacto más probable
            if prob_marcador > max_prob_marcador:
                max_prob_marcador = prob_marcador
                marcador_exacto = (g_local, g_vis)
            
            # Acumular probabilidades de resultados generales
            if g_local > g_vis:
                prob_local += prob_marcador
            elif g_local < g_vis:
                prob_visitante += prob_marcador
            else:
                prob_empate += prob_marcador
                
    return {
        "Prob_Local": round(prob_local * 100, 1),
        "Prob_Empate": round(prob_empate * 100, 1),
        "Prob_Visitante": round(prob_visitante * 100, 1),
        "Marcador_Exacto": marcador_exacto
    }

# ==========================================
# 3. INTERFAZ VISUAL DEL DASHBOARD
# ==========================================
st.title("⚽ Bot Predictor Pro - Mundial 2026")
st.write("Predicciones numéricas exactas basadas en rendimiento histórico, Eliminatorias y Puntuación Elo.")
st.markdown("---")

# Partidos fijos de la Fase de Grupos
PARTIDOS_FASE_DE_GRUPOS = [
    {"Grupo": "Grupo A", "Local": "México", "Visitante": "Sudáfrica"},
    {"Grupo": "Grupo A", "Local": "Francia", "Visitante": "Japón"},
    {"Grupo": "Grupo B", "Local": "Estados Unidos", "Visitante": "Marruecos"},
    {"Grupo": "Grupo B", "Local": "Argentina", "Visitante": "España"},
    {"Grupo": "Grupo C", "Local": "Brasil", "Visitante": "Alemania"},
    {"Grupo": "Grupo C", "Local": "México", "Visitante": "Francia"},
]

df_partidos = pd.DataFrame(PARTIDOS_FASE_DE_GRUPOS)

st.sidebar.header("Filtros del Torneo")
grupos_disponibles = ["Todos"] + list(df_partidos["Grupo"].unique())
grupo_sel = st.sidebar.selectbox("Selecciona un Grupo:", grupos_disponibles)

df_filtrado = df_partidos if grupo_sel == "Todos" else df_partidos[df_partidos["Grupo"] == grupo_sel]

st.subheader("📊 Pronósticos Numéricos de los Partidos")

for index, fila in df_filtrado.iterrows():
    local = fila["Local"]
    visitante = fila["Visitante"]
    grupo = fila["Grupo"]
    
    # Ejecutar el modelo predictivo
    res = predecir_marcador_y_probabilidades(local, visitante)
    ml, mv = res["Marcador_Exacto"]
    
    # Tarjeta del partido
    with st.expander(f"🔮 {grupo}: {local} vs {visitante} — Pronóstico: {local} {ml} - {mv} {visitante}"):
        
        # Mostrar el marcador grande destacado
        st.markdown(f"<h3 style='text-align: center; color: #4A90E2;'>Marcador Más Probable: {local} {ml} - {mv} {visitante}</h3>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label=f"Probabilidad {local}", value=f"{res['Prob_Local']}%")
            st.progress(int(res['Prob_Local']))
        with col2:
            st.metric(label="Probabilidad Empate", value=f"{res['Prob_Empate']}%")
            st.progress(int(res['Prob_Empate']))
        with col3:
            st.metric(label=f"Probabilidad {visitante}", value=f"{res['Prob_Visitante']}%")
            st.progress(int(res['Prob_Visitante']))
