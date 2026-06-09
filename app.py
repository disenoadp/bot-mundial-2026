import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import unicodedata

# Configuración del Dashboard
st.set_page_config(page_title="Bot Predictor Quiniela — Datos Reales", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# BASE DE DATOS HISTÓRICA COMPLETA DE FÚTBOL INTERNACIONAL
URL_HISTORICO_GLOBAL = "https://raw.githubusercontent.com/datasets/football-data/master/data/international-results.csv"

def tokenizar_y_limpiar(texto):
    """Convierte nombres en conjuntos de palabras limpias para el motor de cruce."""
    if not texto or pd.isna(texto) or not isinstance(texto, str):
        return set()
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = texto.lower().strip()
    # Estandarizaciones para asegurar compatibilidad total
    texto = texto.replace("usa", "united states").replace("south korea", "korea republic")
    palabras = texto.replace("-", " ").replace("_", " ").split()
    return set(palabras)

# ==========================================
# 1. CARGA DE BASE DE DATOS HISTÓRICA (CSV)
# ==========================================
@st.cache_data(ttl=86400)
def cargar_base_datos_real():
    """Descarga el CSV global y procesa las estadísticas de rendimiento reales."""
    try:
        df = pd.read_csv(URL_HISTORICO_GLOBAL)
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        
        # Filtrar por fútbol moderno (Año 2010 en adelante)
        df_moderno = df[df['date'].dt.year >= 2010].copy()
        
        rendimiento = {}
        goles_totales = 0
        partidos_totales = 0
        
        for _, fila in df_moderno.iterrows():
            loc = str(fila['home_team']).strip()
            vis = str(fila['away_team']).strip()
            try:
                g_l = int(fila['home_score'])
                g_v = int(fila['away_score'])
            except:
                continue
            
            goles_totales += (g_l + g_v)
            partidos_totales += 1
            
            for equipo, g_anotados, g_recibidos in [(loc, g_l, g_v), (vis, g_v, g_l)]:
                if equipo not in rendimiento:
                    rendimiento[equipo] = {"goles_anotados": 0, "goles_recibidos": 0, "partidos": 0}
                rendimiento[equipo]["goles_anotados"] += g_anotados
                rendimiento[equipo]["goles_recibidos"] += g_recibidos
                rendimiento[equipo]["partidos"] += 1
                
        promedio_global = (goles_totales / (partidos_totales * 2)) if partidos_totales > 0 else 1.35
        
        stats_finales = {}
        for equipo, datos in rendimiento.items():
            pj = datos["partidos"]
            if pj > 0:
                stats_finales[equipo] = {
                    "ofensiva": round((datos["goles_anotados"] / pj) / promedio_global, 2),
                    "defensiva": round((datos["goles_recibidos"] / pj) / promedio_global, 2),
                    "pj": pj,
                    "tokens": tokenizar_y_limpiar(equipo)
                }
        return stats_finales, promedio_global
    except Exception as e:
        st.error(f"Fallo en la lectura automatizada del CSV: {e}")
        return {}, 1.35

# Inicialización obligatoria de la data histórica en memoria
stats_historicas, prom_global = cargar_base_datos_real()

# ==========================================
# 2. MOTOR DE ASOCIACIÓN AUTOMÁTICA
# ==========================================
def buscar_estadisticas_reales(nombre_api):
    if not nombre_api or not isinstance(nombre_api, str):
        return {"nombre_real_csv": "Desconocido", "ofensiva": 1.0, "defensiva": 1.0, "pj": 0}
        
    tokens_api = tokenizar_y_limpiar(nombre_api)
    mejor_coincidencia = None
    max_coincidencias = 0
    
    # Evaluar contra todos los registros indexados del CSV
    for nombre_csv, info in stats_historicas.items():
        coincidencias = len(tokens_api.intersection(info["tokens"]))
        if coincidencias > max_coincidencias:
            max_coincidencias = coincidencias
            mejor_coincidencia = nombre_csv

    # Si se encontró la selección en el CSV, usar sus métricas reales
    if mejor_coincidencia and max_coincidencias > 0:
        res = stats_historicas[mejor_coincidencia].copy()
        res["nombre_real_csv"] = mejor_coincidencia
        return res
            
    # Caso extremo: Equipo sin historial
    return {
        "nombre_real_csv": f"{nombre_api} (Falta Historial)",
        "ofensiva": 1.0,
        "defensiva": 1.0,
        "pj": 0
    }

# ==========================================
# 3. EXTRACCIÓN DESDE LA API DEL MUNDIAL
# ==========================================
@st.cache_data(ttl=3600)
def obtener_partidos_mundial():
    url = f"{BASE_URL}competitions/WC/matches"
    try:
        respuesta = requests.get(url, headers=HEADERS)
        datos = respuesta.json()
        lista_partidos = []
        for match in datos.get("matches", []):
            if not match.get("homeTeam") or not match.get("awayTeam"):
                continue
            lista_partidos.append({
                "Fase": match["stage"].replace("_", " "),
                "Grupo": match.get("group", "Fase Eliminatoria"),
                "Local": match["homeTeam"].get("name", "Por definir"),
                "Visitante": match["awayTeam"].get("name", "Por definir")
            })
        return pd.DataFrame(lista_partidos)
    except:
        return pd.DataFrame()

df_partidos_real = obtener_partidos_mundial()

# ==========================================
# 4. CÁLCULO DE PROBABILIDADES (POISSON)
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    stats_l = buscar_estadisticas_reales(local)
    stats_v = buscar_estadisticas_reales(visitante)
    
    # Ajuste de peso competitivo promedio
    factor_ajuste = 1.25
    goles_esperados_l = stats_l["ofensiva"] * stats_v["defensiva"] * factor_ajuste
    goles_esperados_v = stats_v["ofensiva"] * stats_l["defensiva"] * factor_ajuste
    
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
        "Top_Marcadores": df_m.head(3),
        "Stats_L": stats_l,
        "Stats_V": stats_v
    }

# ==========================================
# 5. INTERFAZ EN STREAMLIT
# ==========================================
st.title("🏆 Bot Predictor Quiniela — Sincronización Automática")
st.write("Cálculo matemático basado en el volumen total de partidos del CSV real.")
st.markdown("---")

if df_partidos_real.empty:
    st.warning("Descargando el calendario oficial de partidos...")
else:
    etapas = sorted(list(df_partidos_real["Fase"].unique()))
    fase_sel = st.sidebar.selectbox("Fase del Concurso:", etapas)
    
    df_filtrado = df_partidos_real[df_partidos_real["Fase"] == fase_sel]
    
    for index, fila in df_filtrado.iterrows():
        local = fila["Local"]
        visitante = fila["Visitante"]
        
        res = calcular_prediccion_concurso(local, visitante)
        g_l, g_v = res["Marcador_Concurso"]
        
        texto_conclusion = f"🎯 GANADOR SUGERIDO: {local}" if res["Tendencia"] == "LOCAL" else (f"🎯 GANADOR SUGERIDO: {visitante}" if res["Tendencia"] == "VISITANTE" else "🎯 RECOMENDACIÓN: Empate Técnico")
        
        with st.expander(f"⚽ {local} vs {visitante}"):
            st.info(f"### 📋 PRONÓSTICO: **{local} {g_l} - {g_v} {visitante}**\n*{texto_conclusion}*")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"**{local}** *(Enlazado con el CSV histórico como: `{res['Stats_L']['nombre_real_csv']}`)*")
                st.write(f"↳ **Partidos reales analizados: {res['Stats_L']['pj']}** | Rendimiento Ataque: {res['Stats_L']['ofensiva']}")
                st.progress(max(0, min(int(res['P_Local']), 100)))
                
                st.write(f"**Empate:** {res['P_Empate']}%")
                st.progress(max(0, min(int(res['P_Empate']), 100)))
                
                st.write(f"**{visitante}** *(Enlazado con el CSV histórico como: `{res['Stats_V']['nombre_real_csv']}`)*")
                st.write(f"↳ **Partidos reales analizados: {res['Stats_V']['pj']}** | Rendimiento Ataque: {res['Stats_V']['ofensiva']}")
                st.progress(max(0, min(int(res['P_Visitante']), 100)))
            with col2:
                st.write("**🎲 Escenarios más probables:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    st.write(f"• {local} {ml} - {mv} {visitante} ({round(m_fila['prob']*100,1)}%)")
