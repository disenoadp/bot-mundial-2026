import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import pytz

# Configuración del Dashboard
st.set_page_config(page_title="Bot Predictor 100% Automatizado", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# URL pública con el Coeficiente de Fuerza Internacional basado en el Ranking FIFA actual
# Mantenido de forma abierta para que el bot calcule el poder inicial sin datos hardcoded en el script
URL_RANKING_REMOTO = "https://raw.githubusercontent.com/martivo/datasets/main/world_cup_powers.csv"

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
# 2. CÁLCULO DE PODER COMBINANDO RANKING ONLINE + GOLES EN VIVO
# ==========================================
@st.cache_data(ttl=3600)
def calcular_fuerza_equipos_hibrido():
    """
    Lee los coeficientes iniciales desde un repositorio de datos de fútbol en internet
    y los combina dinámicamente con los goles que vayan ocurriendo en el torneo actual.
    """
    # 1. Cargar la fuerza base histórica desde internet (Evita datos manuales en el código)
    try:
        df_ranking = pd.read_csv(URL_RANKING_REMOTO)
        # Convertimos el CSV online en un diccionario de consulta rápida
        stats_online = df_ranking.set_index('team')[['ofensiva', 'defensiva']].to_dict(orient='index')
    except:
        # Red de seguridad vacía si falla la conexión al CSV externo
        stats_online = {}

    # 2. Consultar partidos en vivo para actualizar el momento actual
    url_partidos = f"{BASE_URL}competitions/WC/matches"
    rendimiento_vivo = {}
    fuerza_base = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
    
    try:
        respuesta = requests.get(url_partidos, headers=HEADERS)
        matches = respuesta.json().get("matches", [])
        
        for m in matches:
            if m["status"] == "FINISHED":
                loc = m["homeTeam"]["name"]
                vis = m["awayTeam"]["name"]
                g_l = m["score"]["fullTime"]["home"]
                g_v = m["score"]["fullTime"]["away"]
                
                if g_l is not None and g_v is not None:
                    if loc not in rendimiento_vivo: rendimiento_vivo[loc] = fuerza_base.copy()
                    if vis not in rendimiento_vivo: rendimiento_vivo[vis] = fuerza_base.copy()
                    
                    rendimiento_vivo[loc]["goles_anotados"] += g_l
                    rendimiento_vivo[loc]["goles_recibidos"] += g_v
                    rendimiento_vivo[loc]["partidos"] += 1
                    rendimiento_vivo[vis]["goles_anotados"] += g_v
                    rendimiento_vivo[vis]["goles_recibidos"] += g_l
                    rendimiento_vivo[vis]["partidos"] += 1
    except:
        pass # Si la API de goles falla temporalmente, nos quedamos con el ranking base

    # 3. Fusión matemática final
    stats_finales = {}
    # Unificamos todos los equipos del ecosistema
    todos_los_equipos = set(list(stats_online.keys()) + list(rendimiento_vivo.keys()))
    
    for equipo in todos_los_equipos:
        # Valores por defecto si el equipo no figura en el ranking histórico online
        base = stats_online.get(equipo, {"ofensiva": 1.2, "defensiva": 1.1})
        
        if equipo in rendimiento_vivo and rendimiento_vivo[equipo]["partidos"] > 0:
            pj = rendimiento_vivo[equipo]["partidos"]
            prom_favor = rendimiento_vivo[equipo]["goles_anotados"] / pj
            prom_contra = rendimiento_vivo[equipo]["goles_recibidos"] / pj
            
            # El poder actual es un promedio equilibrado entre su historia internacional y su presente en el torneo
            stats_finales[equipo] = {
                "ofensiva": (base["ofensiva"] + prom_favor) / 2,
                "defensiva": (base["defensiva"] + prom_contra) / 2
            }
        else:
            stats_finales[equipo] = base

    return stats_finales

df_partidos_real = obtener_partidos_mundial()
stats_automaticas = calcular_fuerza_equipos_hibrido()

# ==========================================
# 3. CEREBRO PREDICTOR MATEMÁTICO (POISSON)
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    default = {"ofensiva": 1.1, "defensiva": 1.1}
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
st.title("🏆 Bot Predictor Quiniela - Datos 100% Online")
st.write("Cálculo automatizado: Lee la fuerza inicial histórica de la nube y le suma el rendimiento en vivo.")
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
