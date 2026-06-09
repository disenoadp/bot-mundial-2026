import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import pytz

# Configuración del Dashboard
st.set_page_config(page_title="Bot Inteligente Quiniela", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# ==========================================
# DATOS HISTÓRICOS DE RESPALDO (Fuerza Real Inicial)
# ==========================================
RENDIMIENTO_HISTORICO_INICIAL = {
    "Argentina": {"ofensiva": 2.3, "defensiva": 0.6},
    "France": {"ofensiva": 2.2, "defensiva": 0.7},
    "Spain": {"ofensiva": 2.1, "defensiva": 0.7},
    "Brazil": {"ofensiva": 2.0, "defensiva": 0.8},
    "Germany": {"ofensiva": 1.9, "defensiva": 0.9},
    "Portugal": {"ofensiva": 1.9, "defensiva": 0.9},
    "Netherlands": {"ofensiva": 1.8, "defensiva": 1.0},
    "England": {"ofensiva": 1.8, "defensiva": 0.9},
    "Japan": {"ofensiva": 1.7, "defensiva": 0.9},
    "Morocco": {"ofensiva": 1.5, "defensiva": 0.8},
    "USA": {"ofensiva": 1.5, "defensiva": 1.0},
    "Mexico": {"ofensiva": 1.4, "defensiva": 1.1},
    "South Korea": {"ofensiva": 1.4, "defensiva": 1.1},
    "Czechia": {"ofensiva": 1.3, "defensiva": 1.1},
    "Canada": {"ofensiva": 1.3, "defensiva": 1.2},
    "South Africa": {"ofensiva": 1.1, "defensiva": 1.3},
    "Bosnia-Herzegovina": {"ofensiva": 1.1, "defensiva": 1.3},
}

# ==========================================
# 1. DESCARGA EN VIVO DEL CALENDARIO OFICIAL
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
# 2. PROCESAMIENTO HÍBRIDO (HISTÓRICO + EN VIVO)
# ==========================================
@st.cache_data(ttl=3600)
def calcular_estadisticas_automaticas():
    url = f"{BASE_URL}competitions/WC/matches"
    rendimiento_vivo = {}
    
    try:
        respuesta = requests.get(url, headers=HEADERS)
        datos = respuesta.json()
        matches = datos.get("matches", [])
        
        goles_totales, partidos_totales = 0, 0
        
        for m in matches:
            if m["status"] == "FINISHED":
                loc = m["homeTeam"]["name"]
                vis = m["awayTeam"]["name"]
                g_l = m["score"]["fullTime"]["home"]
                g_v = m["score"]["fullTime"]["away"]
                
                goles_totales += (g_l + g_v)
                partidos_totales += 1
                
                for equipo in [loc, vis]:
                    if equipo not in rendimiento_vivo:
                        rendimiento_vivo[equipo] = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
                
                rendimiento_vivo[loc]["goles_anotados"] += g_l
                rendimiento_vivo[loc]["goles_recibidos"] += g_v
                rendimiento_vivo[loc]["partidos"] += 1
                rendimiento_vivo[vis]["goles_anotados"] += g_v
                rendimiento_vivo[vis]["goles_recibidos"] += g_l
                rendimiento_vivo[vis]["partidos"] += 1
        
        # Combinar datos históricos con los de en vivo
        stats_finales = {}
        
        # Mapeamos todos los equipos posibles utilizando la lista histórica como base inicial
        for equipo, datos_hist en RENDIMIENTO_HISTORICO_INICIAL.items():
            if equipo in rendimiento_vivo and rendimiento_vivo[equipo]["partidos"] > 0:
                # Si ya hay partidos en vivo, promedia el historial con la actualidad
                pj = rendimiento_vivo[equipo]["partidos"]
                prom_favor_vivo = rendimiento_vivo[equipo]["goles_anotados"] / pj
                prom_contra_vivo = rendimiento_vivo[equipo]["goles_recibidos"] / pj
                
                stats_finales[equipo] = {
                    "ofensiva": (datos_hist["ofensiva"] + prom_favor_vivo) / 2,
                    "defensiva": (datos_hist["defensiva"] + prom_contra_vivo) / 2
                }
            else:
                # Si no ha jugado en el torneo, usa su fuerza histórica real
                stats_finales[equipo] = datos_hist
                
        return stats_finales
    except:
        return RENDIMIENTO_HISTORICO_INICIAL

df_partidos_real = obtener_partidos_mundial()
stats_automaticas = calcular_estadisticas_automaticas()

# ==========================================
# 3. CEREBRO PREDICTOR MATEMÁTICO
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    default = {"ofensiva": 1.2, "defensiva": 1.2}
    stats_l = stats_automaticas.get(local, default)
    stats_v = stats_automaticas.get(visitante, default)
    
    goles_esperados_l = stats_l["ofensiva"] * stats_v["defensiva"]
    goles_esperados_v = stats_v["ofensiva"] * stats_l["defensiva"]
    
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
st.title("🏆 Bot Predictor Quiniela - Modo Híbrido")
st.write("Predicciones iniciales basadas en Ranking/Eliminatorias, actualizadas automáticamente con los goles del torneo en tiempo real.")
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
                st.write("**📊 Probabilidades de resultado:**")
                st.write(f"Victoria {local}: {res['P_Local']}%")
                st.progress(int(res['P_Local']))
                st.write(f"Empate: {res['P_Empate']}%")
                st.progress(int(res['P_Empate']))
                st.write(f"Victoria {visitante}: {res['P_Visitante']}%")
                st.progress(int(res['P_Visitante']))
            with col2:
                st.write("**🎲 Top 3 marcadores:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    st.write(f"• {local} {ml} - {mv} {visitante} ({round(m_fila['prob']*100,1)}%)")
