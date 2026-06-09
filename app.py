import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import pytz

# Configuración del Dashboard
st.set_page_config(page_title="Bot Predictor Histórico Real", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# Base de datos global y pública de partidos internacionales (Actualizada constantemente)
URL_HISTORICO_GLOBAL = "https://raw.githubusercontent.com/martivo/datasets/main/international_results.csv"

# ==========================================
# 1. DESCARGA EN VIVO DEL CALENDARIO OFICIAL (API)
# ==========================================
@st.cache_data(ttl=3600)
def obtener_partidos_mundial():
    url = f"{BASE_URL}competitions/WC/matches"
    try:
        respuesta = requests.get(url, headers=HEADERS)
        datos = respuesta.json()
        lista_partidos = []
        for match in datos.get("matches", []):
            lista_partidos.append({
                "Fase": match["stage"].replace("_", " "),
                "Grupo": match.get("group", "Fase Eliminatoria"),
                "Local": match["homeTeam"]["name"],
                "Visitante": match["awayTeam"]["name"],
                "Fecha_UTC": match["utcDate"],
                "Estado": match["status"]
            })
        return pd.DataFrame(lista_partidos)
    except:
        return pd.DataFrame()

# ==========================================
# 2. PROCESAMIENTO DEL DESEMPEÑO HISTÓRICO REAL (ONLINE)
# ==========================================
@st.cache_data(ttl=86400) # Se guarda por 24 horas porque el historial no cambia a cada minuto
def calcular_fuerza_desde_historico():
    """
    Descarga el archivo histórico de fútbol internacional, analiza los goles reales
    de los últimos años para cada selección y calcula su poder estadístico.
    """
    try:
        # Descarga la base de datos de partidos de internet
        df = pd.read_csv(URL_HISTORICO_GLOBAL)
        
        # Filtramos para usar solo partidos modernos (ej. desde el año 2018 en adelante)
        # Esto asegura que evaluamos el rendimiento actual y no lo que pasó en 1950
        df['date'] = pd.to_datetime(df['date'])
        df_moderno = df[df['date'].dt.year >= 2018]
        
        rendimiento = {}
        fuerza_base = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
        
        goles_totales = 0
        partidos_totales = 0
        
        for _, fila in df_moderno.iterrows():
            loc = fila['home_team']
            vis = fila['away_team']
            g_l = int(fila['home_score'])
            g_v = int(fila['away_score'])
            
            goles_totales += (g_l + g_v)
            partidos_totales += 1
            
            if loc not in rendimiento: rendimiento[loc] = fuerza_base.copy()
            if vis not in rendimiento: rendimiento[vis] = fuerza_base.copy()
            
            rendimiento[loc]["goles_anotados"] += g_l
            rendimiento[loc]["goles_recibidos"] += g_v
            rendimiento[loc]["partidos"] += 1
            
            rendimiento[vis]["goles_anotados"] += g_v
            rendimiento[vis]["goles_recibidos"] += g_l
            rendimiento[vis]["partidos"] += 1
            
        promedio_global = (goles_totales / (partidos_totales * 2)) if partidos_totales > 0 else 1.3
        
        stats_finales = {}
        for equipo, datos_e in rendimiento.items():
            pj = datos_e["partidos"]
            if pj > 0:
                stats_finales[equipo] = {
                    "ofensiva": round((datos_e["goles_anotados"] / pj) / promedio_global, 2),
                    "defensiva": round((datos_e["goles_recibidos"] / pj) / promedio_global, 2)
                }
        return stats_finales
    except:
        # Red de seguridad si el enlace falla
        return {}

df_partidos_real = obtener_partidos_mundial()
stats_historicas = calcular_fuerza_desde_historico()

# ==========================================
# 3. CEREBRO PREDICTOR MATEMÁTICO (POISSON)
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    # Valores promedio si un país no tiene partidos registrados desde 2018
    default_l = {"ofensiva": 1.2, "defensiva": 1.0}
    default_v = {"ofensiva": 1.1, "defensiva": 1.1}
    
    # Mapeo de nombres (La API usa inglés, resolvemos compatibilidad básica si es necesario)
    stats_l = stats_historicas.get(local, default_l)
    stats_v = stats_historicas.get(visitante, default_v)
    
    # Cruce directo de rendimiento histórico: Ataque de uno contra defensa del otro
    goles_esperados_l = stats_l["ofensiva"] * stats_v["defensiva"] * 1.35
    goles_esperados_v = stats_v["ofensiva"] * stats_l["defensiva"] * 1.35
    
    prob_local, prob_empate, prob_visitante = 0, 0, 0
    todos_los_marcadores = []
    
    for g_local in range(6):
        for g_vis in range(6):
            p_l = poisson.pmf(g_local, goles_esperados_l)
            p_v = poisson.pmf(g_vis, goles_esperados_v)
            prob_marcador = p_l * p_v
            
            tipo = "LOCAL" if g_local > g_vis else ("VISITANTE" if g_local < g_vis else "EMPATE")
            todos_los_marcadores.append({"marcador": (g_local, g_vis), "prob": prob_marcador, "tipo": tipo})
            
            if tipo == "LOCAL": prob_local += prob_marcador
            elif tipo == "VISITANTE": prob_visitante += prob_marcador
            else: prob_empate += prob_marcador

    df_m = pd.DataFrame(todos_los_marcadores).sort_values(by="prob", ascending=False)
    porcentajes = {"LOCAL": prob_local, "EMPATE": prob_empate, "VISITANTE": prob_visitante}
    tendencia_ganadora = max(porcentajes, key=porcentajes.get)
    
    marcador_coherente_fila = df_m[df_m["tipo"] == tendencia_ganadora].iloc[0]
    return {
        "P_Local": round(prob_local * 100, 1),
        "P_Empate": round(prob_empate * 100, 1),
        "P_Visitante": round(prob_visitante * 100, 1),
        "Tendencia": tendencia_ganadora,
        "Marcador_Concurso": marcador_coherente_fila["marcador"],
        "Top_Marcadores": df_m.head(3)
    }

# ==========================================
# 4. INTERFAZ VISUAL AUTOMÁTICA
# ==========================================
st.title("🏆 Bot Predictor Quiniela - Historial Científico")
st.write("Análisis basado en el registro real de partidos internacionales oficiales jugados desde el 2018.")
st.markdown("---")

if df_partidos_real.empty:
    st.warning("Conectando con los servidores de la API...")
else:
    st.sidebar.header("Etapa del Torneo")
    etapas = sorted(list(df_partidos_real["Fase"].unique()))
    fase_sel = st.sidebar.selectbox("Selecciona la Fase:", etapas)
    
    df_filtrado = df_partidos_real[df_partidos_real["Fase"] == fase_sel]
    
    for index, fila in df_filtrado.iterrows():
        local = fila["Local"]
        visitante = fila["Visitante"]
        
        try:
            fecha_utc = datetime.strptime(fila["Fecha_UTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
            fecha_local = fecha_utc.astimezone(pytz.timezone('America/Lima'))
            fecha_str = fecha_local.strftime('%d/%m/%Y - %H:%M')
        except:
            fecha_str = "Fecha no disponible"
        
        res = calcular_prediccion_concurso(local, visitante)
        g_l, g_v = res["Marcador_Concurso"]
        
        texto_conclusion = f"🎯 GANADOR: {local}" if res["Tendencia"] == "LOCAL" else (f"🎯 GANADOR: {visitante}" if res["Tendencia"] == "VISITANTE" else "🎯 RESULTADO: Empate")
        
        with st.expander(f"📅 {fecha_str} hs (Hora Perú) | {local} vs {visitante}"):
            st.info(f"### 📋 RECOMENDACIÓN:\n**{texto_conclusion}** con un marcador de **{local} {g_l} - {g_v} {visitante}**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**📊 Probabilidades basadas en datos históricos reales:**")
                st.write(f"Victoria {local}: {res['P_Local']}%")
                st.progress(int(res['P_Local']))
                st.write(f"Empate: {res['P_Empate']}%")
                st.progress(int(res['P_Empate']))
                st.write(f"Victoria {visitante}: {res['P_Visitante']}%")
                st.progress(int(res['P_Visitante']))
            with col2:
                st.write("**🎲 Marcadores más probables:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    st.write(f"• {local} {ml} - {mv} {visitante} ({round(m_fila['prob']*100,1)}%)")
