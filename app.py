import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import pytz

# Configuración del Dashboard
st.set_page_config(page_title="Bot Predictor 100% Online", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

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
# 2. CÁCULO DE PODER BASADO 100% EN DATOS ONLINE (ELIMINATORIAS + MUNDIAL)
# ==========================================
@st.cache_data(ttl=3600)
def calcular_fuerza_equipos_online():
    """
    Consulta las estadísticas de los partidos oficiales guardados en la API 
    para determinar el poder real de ataque y defensa sin intervención humana.
    """
    rendimiento = {}
    goles_totales = 0
    partidos_totales = 0
    fuerza_base = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
    
    # Intentamos leer el historial de partidos de la competición para evaluar rendimiento
    url_partidos = f"{BASE_URL}competitions/WC/matches"
    try:
        respuesta = requests.get(url_partidos, headers=HEADERS)
        matches = respuesta.json().get("matches", [])
        
        for m in matches:
            # Procesamos todos los partidos completados del proceso (incluyendo clasificatorios si están en el histórico del id)
            if m["status"] == "FINISHED":
                loc = m["homeTeam"]["name"]
                vis = m["awayTeam"]["name"]
                g_l = m["score"]["fullTime"]["home"]
                g_v = m["score"]["fullTime"]["away"]
                
                # Control de nulos de seguridad de la API
                if g_l is not None and g_v is not None:
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
                    
        # Promedio global de goles del ecosistema FIFA
        promedio_global = (goles_totales / (partidos_totales * 2)) if partidos_totales > 0 else 1.35
        
        # Convertir a factores métricos de Poisson
        stats_finales = {}
        for equipo, datos_e in rendimiento.items():
            pj = datos_e["partidos"]
            if pj > 0:
                stats_finales[equipo] = {
                    "ofensiva": (datos_e["goles_anotados"] / pj) / promedio_global,
                    "defensiva": (datos_e["goles_recibidos"] / pj) / promedio_global
                }
        return stats_finales
    except:
        return {}

df_partidos_real = obtener_partidos_mundial()
stats_automaticas = calcular_fuerza_equipos_online()

# ==========================================
# 3. CEREBRO PREDICTOR MATEMÁTICO
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    # Si un equipo es nuevo y no tiene historial absoluto registrado, se le asigna poder neutro (1.0)
    # Evitando que el código colapse y garantizando un punto de partida justo
    default = {"ofensiva": 1.0, "defensiva": 1.0}
    stats_l = stats_automaticas.get(local, default)
    stats_v = stats_automaticas.get(visitante, default)
    
    # Ajuste dinámico de goles esperados promedio por partido en mundiales (base 1.38)
    goles_esperados_l = stats_l["ofensiva"] * stats_v["defensiva"] * 1.38
    goles_esperados_v = stats_v["ofensiva"] * stats_l["defensiva"] * 1.38
    
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
st.title("🏆 Bot Predictor Quiniela - Cero Datos Manuales")
st.write("Análisis estadístico puro. El bot procesa el historial clasificatorio FIFA en línea de cada selección para definir sus niveles.")
st.markdown("---")

if df_partidos_real.empty:
    st.warning("Conectando con los servidores de la API de Fútbol...")
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
            st.info(f"### 📋 RECOMENDACIÓN PARA EL CONCURSO:\n**{texto_conclusion}** con un marcador de **{local} {g_l} - {g_v} {visitante}**")
            
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
