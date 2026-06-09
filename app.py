import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import unicodedata

# Configuración del Dashboard
st.set_page_config(page_title="Bot Predictor Quiniela — Producción", page_icon="⚽", layout="wide")

API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# Url directa al repositorio original de resultados internacionales
URL_HISTORICO_GLOBAL = "https://raw.githubusercontent.com/datasets/football-data/master/data/international-results.csv"

def mapear_nombre_comun(texto):
    """Mapea las discrepancias de nombres entre la API y el CSV histórico."""
    if not texto:
        return ""
    # Conversión a minúsculas y limpieza básica
    t = texto.lower().strip()
    
    # Diccionario de homologación API <=> CSV
    diccionario_paises = {
        "czechia": "czech republic",
        "czech republic": "czech republic",
        "south korea": "korea republic",
        "korea republic": "korea republic",
        "usa": "united states",
        "united states": "united states",
        "mexico": "mexico",
        "south africa": "south africa",
        "canada": "canada",
        "bosnia and herzegovina": "bosnia-herzegovina",
        "bosnia-herzegovina": "bosnia-herzegovina"
    }
    
    return diccionario_paises.get(t, t)

def tokenizar_y_limpiar(texto):
    if not texto or pd.isna(texto) or not isinstance(texto, str):
        return set()
    # Quitar acentos y caracteres especiales
    texto = unicodedata.normalize('NFKD', texto).encode('ASCII', 'ignore').decode('utf-8')
    texto = mapear_nombre_comun(texto)
    palabras = texto.replace("-", " ").replace("_", " ").split()
    return set(palabras)

# ==========================================
# 1. CARGA FORZADA DE DATOS (SIN CACHÉ CORRUPTA)
# ==========================================
def cargar_base_datos_real_forzado():
    """Descarga el CSV directamente en cada reinicio para asegurar consistencia."""
    try:
        df = pd.read_csv(URL_HISTORICO_GLOBAL)
        df.columns = [col.lower() for col in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        
        # Filtramos desde el 2015 para máxima fidelidad con plantillas modernas
        df_moderno = df[df['date'].dt.year >= 2015].copy()
        
        rendimiento = {}
        goles_totales = 0
        partidos_totales = 0
        
        for _, fila in df_moderno.iterrows():
            loc = mapear_nombre_comun(str(fila['home_team']))
            vis = mapear_nombre_comun(str(fila['away_team']))
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
        return stats_finales, promedio_global, True
    except Exception as e:
        return {}, 1.35, False

# Inicialización limpia de variables de control
stats_historicas, prom_global, csv_cargado_exito = cargar_base_datos_real_forzado()

# ==========================================
# 2. MOTOR DE BÚSQUEDA SEMÁNTICA
# ==========================================
def buscar_estadisticas_reales(nombre_api):
    if not nombre_api or not isinstance(nombre_api, str):
        return {"nombre_real_csv": "Desconocido", "ofensiva": 1.0, "defensiva": 1.0, "pj": 0}
        
    nombre_limpio_api = mapear_nombre_comun(nombre_api)
    tokens_api = tokenizar_y_limpiar(nombre_limpio_api)
    
    # Intento 1: Coincidencia directa por llave del diccionario
    if nombre_limpio_api in stats_historicas:
        res = stats_historicas[nombre_limpio_api].copy()
        res["nombre_real_csv"] = nombre_limpio_api
        return res
        
    # Intento 2: Intersección de tokens de respaldo
    mejor_coincidencia = None
    max_coincidencias = 0
    for nombre_csv, info in stats_historicas.items():
        coincidencias = len(tokens_api.intersection(info["tokens"]))
        if coincidencias > max_coincidencias:
            max_coincidencias = coincidencias
            mejor_coincidencia = nombre_csv

    if mejor_coincidencia and max_coincidencias > 0:
        res = stats_historicas[mejor_coincidencia].copy()
        res["nombre_real_csv"] = mejor_coincidencia
        return res
            
    # Intento 3: Asignación por defecto calculada si el equipo es completamente nuevo
    return {
        "nombre_real_csv": f"{nombre_api} (Perfil Genérico)",
        "ofensiva": 1.1,
        "defensiva": 1.1,
        "pj": 12 # Simulación de partidos base para evitar el 0
    }

# ==========================================
# 3. CONEXIÓN API EN VIVO
# ==========================================
@st.cache_data(ttl=1800)
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
# 4. MODELO MATEMÁTICO POISSON
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    stats_l = buscar_estadisticas_reales(local)
    stats_v = buscar_estadisticas_reales(visitante)
    
    # Constante de ajuste de goles para torneos cortos de alta intensidad
    factor_ajuste = 1.22
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
# 5. RENDERIZADO DEL DASHBOARD
# ==========================================
st.title("🏆 Bot Predictor Quiniela — Versión de Producción")
st.write("Cruce en caliente API mundial ⇄ Historial CSV sin almacenamiento de errores.")
st.markdown("---")

# Muestra el estado del cargador del archivo CSV en la cabecera
if csv_cargado_exito:
    st.sidebar.success(f" Base de datos cargada. {len(stats_historicas)} selecciones listas.")
else:
    st.sidebar.error("⚠️ Error crítico al descargar el CSV de GitHub. Usando contingencia genérica.")

if df_partidos_real.empty:
    st.warning("Descargando partidos desde la API de Fútbol...")
else:
    etapas = sorted(list(df_partidos_real["Fase"].unique()))
    fase_sel = st.sidebar.selectbox("Selecciona la Fase:", etapas)
    
    df_filtrado = df_partidos_real[df_partidos_real["Fase"] == fase_sel]
    
    for index, fila in df_filtrado.iterrows():
        local = fila["Local"]
        visitante = fila["Visitante"]
        
        res = calcular_prediccion_concurso(local, visitante)
        g_l, g_v = res["Marcador_Concurso"]
        
        texto_conclusion = f"🎯 SUGERENCIA: Gana {local}" if res["Tendencia"] == "LOCAL" else (f"🎯 SUGERENCIA: Gana {visitante}" if res["Tendencia"] == "VISITANTE" else "🎯 SUGERENCIA: Empate")
        
        with st.expander(f"⚽ {local} vs {visitante} (Predicción Dinámica)"):
            st.info(f"### **Pronóstico Sugerido: {local} {g_l} - {g_v} {visitante}**\n*{texto_conclusion}*")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write(f"📊 **{local}** — Partidos analizados: `{res['Stats_L']['pj']}` | Factor Ofensivo: `{res['Stats_L']['ofensiva']}`")
                st.write(f"Probabilidad de Victoria: {res['P_Local']}%")
                st.progress(max(0, min(int(res['P_Local']), 100)))
                
                st.write(f"⚖️ **Empate:** {res['P_Empate']}%")
                st.progress(max(0, min(int(res['P_Empate']), 100)))
                
                st.write(f"📊 **{visitante}** — Partidos analizados: `{res['Stats_V']['pj']}` | Factor Ofensivo: `{res['Stats_V']['ofensiva']}`")
                st.write(f"Probabilidad de Victoria: {res['P_Visitante']}%")
                st.progress(max(0, min(int(res['P_Visitante']), 100)))
                
            with col2:
                st.write("**🎲 Top 3 Marcadores Exactos:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    st.write(f"• {local} {ml} - {mv} {visitante} ({round(m_fila['prob']*100,1)}%)")
