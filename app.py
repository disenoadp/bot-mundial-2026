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
            # IMPORTANTE: Captura tanto fase de grupos como las llaves finales automáticamente
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
# 2. PROCESAMIENTO AUTOMÁTICO DE ESTADÍSTICAS REALES
# ==========================================
@st.cache_data(ttl=3600)
def calcular_estadisticas_automaticas():
    """
    Analiza todos los partidos jugados en el torneo actual para calcular
    las fuerzas reales de ataque y defensa de cada selección en internet.
    """
    url = f"{BASE_URL}competitions/WC/matches"
    # Diccionario base por si un equipo no ha jugado ningún partido aún
    fuerza_defecto = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
    rendimiento = {}
    
    try:
        respuesta = requests.get(url, headers=HEADERS)
        datos = respuesta.json()
        matches = datos.get("matches", [])
        
        goles_totales = 0
        partidos_totales = 0
        
        # 1. Acumular goles reales anotados y recibidos en vivo
        for m in matches:
            if m["status"] == "FINISHED":
                loc = m["homeTeam"]["name"]
                vis = m["awayTeam"]["name"]
                g_l = m["score"]["fullTime"]["home"]
                g_v = m["score"]["fullTime"]["away"]
                
                goles_totales += (g_l + g_v)
                partidos_totales += 1
                
                if loc not in rendimiento: rendimiento[loc] = fuerza_defecto.copy()
                if vis not in rendimiento: rendimiento[vis] = fuerza_defecto.copy()
                
                rendimiento[loc]["goles_anotados"] += g_l
                rendimiento[loc]["goles_recibidos"] += g_v
                rendimiento[loc]["partidos"] += 1
                
                rendimiento[vis]["goles_anotados"] += g_v
                rendimiento[vis]["goles_recibidos"] += g_l
                rendimiento[vis]["partidos"] += 1
        
        # Calcular el promedio de goles global del torneo (fórmula de Poisson)
        promedio_global = (goles_totales / (partidos_totales * 2)) if partidos_totales > 0 else 1.3
        
        # 2. Traducir goles a métricas científicas de Ofensiva y Defensiva
        stats_finales = {}
        for equipo, datos_e in rendimiento.items():
            pj = datos_e["partidos"]
            if pj > 0:
                prom_favor = datos_e["goles_anotados"] / pj
                prom_contra = datos_e["goles_recibidos"] / pj
                
                # Fuerza relativa respecto al promedio de todo el torneo
                stats_finales[equipo] = {
                    "ofensiva": prom_favor / promedio_global,
                    "defensiva": prom_contra / promedio_global
                }
        return stats_finales
    except:
        return {}

df_partidos_real = obtener_partidos_mundial()
stats_automaticas = calcular_estadisticas_automaticas()

# ==========================================
# 3. CEREBRO PREDICTOR MATEMÁTICO
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    # Si la API no tiene registros del equipo aún, asigna valores neutros (fuerza equilibrada)
    default = {"ofensiva": 1.0, "defensiva": 1.0}
    stats_l = stats_automaticas.get(local, default)
    stats_v = stats_automaticas.get(visitante, default)
    
    # Goles promedio esperados puros basados en rendimiento real del torneo
    goles_esperados_l = stats_l["ofensiva"] * stats_v["defensiva"] * 1.3
    goles_esperados_v = stats_v["ofensiva"] * stats_l["defensiva"] * 1.3
    
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
st.title("🏆 Bot Predictor 100% Automatizado")
st.write("Estadísticas de ataque y defensa calculadas dinámicamente según los goles reales del torneo.")
st.markdown("---")

if df_partidos_real.empty:
    st.warning("Conectando con los servidores de la FIFA...")
else:
    # Filtro inteligente por etapa del torneo (se actualizará solo a Octavos, Cuartos, etc.)
    st.sidebar.header("Etapa del Torneo")
    etapas = sorted(list(df_partidos_real["Fase"].unique()))
    fase_sel = st.sidebar.selectbox("Selecciona la Fase:", etapas)
    
    df_filtrado = df_partidos_real[df_partidos_real["Fase"] == fase_sel]
    
    for index, fila in df_filtrado.iterrows():
        local = fila["Local"]
        visitante = fila["Visitante"]
        
        # Ajustar fecha y hora a Perú
        fecha_utc = datetime.strptime(fila["Fecha_UTC"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=pytz.utc)
        fecha_local = fecha_utc.astimezone(pytz.timezone('America/Lima'))
        
        res = calcular_prediccion_concurso(local, visitante)
        g_l, g_v = res["Marcador_Concurso"]
        
        texto_conclusion = f"🎯 GANADOR: {local}" if res["Tendencia"] == "LOCAL" else (f"🎯 GANADOR: {visitante}" if res["Tendencia"] == "VISITANTE" else "🎯 RESULTADO: Empate")
        
        with st.expander(f"📅 {fecha_local.strftime('%d/%m/%Y')} - {fecha_local.strftime('%H:%M')} hs | {local} vs {visitante}"):
            st.info(f"### 📋 RECOMENDACIÓN:\n**{texto_conclusion}** con un marcador de **{local} {g_l} - {g_v} {visitante}**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**📊 Probabilidades calculadas por rendimiento en el torneo:**")
                st.write(f"Victoria {local}: {res['P_Local']}%")
                st.progress(int(res['P_Local']))
                st.write(f"Empate: {res['P_Empate']}%")
                st.progress(int(res['P_Empate']))
                st.write(f"Victoria {visitante}: {res['P_Visitante']}%")
                st.progress(int(res['P_Visitante']))
            with col2:
                st.write("**🎲 Fórmulas alternativas:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    st.write(f"• {local} {ml} - {mv} {visitante} ({round(m_fila['prob']*100,1)}%)")
