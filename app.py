import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from datetime import datetime
import pytz

# Configuración de la interfaz del Dashboard
st.set_page_config(page_title="Bot Quiniela Mundial 2026", page_icon="⚽", layout="wide")

# Recuperar el Token seguro de la API desde los Secrets de Streamlit
API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
HEADERS = {"X-Auth-Token": API_TOKEN}

# ==========================================
# 1. CONEXIÓN EN LÍNEA Y DESCARGA DE DATA REAL
# ==========================================
@st.cache_data(ttl=3600)
def obtener_partidos_mundial():
    url = f"{BASE_URL}competitions/WC/matches"
    try:
        respuesta = requests.get(url, headers=HEADERS)
        datos = respuesta.json()
        lista_partidos = []
        for match in datos.get("matches", []):
            if match["stage"] == "GROUP_STAGE":
                # Capturar el horario en formato crudo UTC de internet
                fecha_utc_str = match["utcDate"]
                
                # Convertir el texto de la API en un objeto de tiempo manipulable
                fecha_utc = datetime.strptime(fecha_utc_str, "%Y-%m-%dT%H:%M:%SZ")
                fecha_utc = pytz.utc.localize(fecha_utc)
                
                # CONFIGURACIÓN EXACTA PARA LIMA, PERÚ (UTC-5)
                zona_local = pytz.timezone('America/Lima')
                fecha_local = fecha_utc.astimezone(zona_local)
                
                # Formatear la fecha y hora con diseño amigable en español
                fecha_formateada = fecha_local.strftime("%d/%m/%Y")
                hora_formateada = fecha_local.strftime("%H:%M")

                lista_partidos.append({
                    "Grupo": match.get("group", "Fase de Grupos"),
                    "Local": match["homeTeam"]["name"],
                    "Visitante": match["awayTeam"]["name"],
                    "Fecha": fecha_formateada,
                    "Hora": hora_formateada
                })
        return pd.DataFrame(lista_partidos)
    except:
        # Respaldo seguro por si hay microcortes en el servidor de fútbol
        return pd.DataFrame([{"Grupo": "Grupo A", "Local": "Mexico", "Visitante": "South Africa", "Fecha": "11/06/2026", "Hora": "15:00"}])

@st.cache_data
def obtener_estadisticas_actualizadas():
    return {
        "Argentina": {"ofensiva": 2.3, "defensiva": 0.6},
        "France": {"ofensiva": 2.2, "defensiva": 0.7},
        "Spain": {"ofensiva": 2.1, "defensiva": 0.7},
        "Brazil": {"ofensiva": 2.0, "defensiva": 0.8},
        "Mexico": {"ofensiva": 1.4, "defensiva": 1.1},
        "USA": {"ofensiva": 1.5, "defensiva": 1.0},
        "Germany": {"ofensiva": 1.9, "defensiva": 0.9},
        "Japan": {"ofensiva": 1.7, "defensiva": 0.9},
        "Morocco": {"ofensiva": 1.5, "defensiva": 0.8},
        "South Africa": {"ofensiva": 1.1, "defensiva": 1.3},
    }

df_partidos_real = obtener_partidos_mundial()
stats_dinamicas = obtener_estadisticas_actualizadas()

# ==========================================
# 2. CEREBRO PREDICTOR COHERENTE (CONCURSO)
# ==========================================
def calcular_prediccion_concurso(local, visitante):
    default = {"ofensiva": 1.3, "defensiva": 1.2}
    stats_l = stats_dinamicas.get(local, default)
    stats_v = stats_dinamicas.get(visitante, default)
    
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
            
            todos_los_marcadores.append({
                "marcador": (g_local, g_vis),
                "prob": prob_marcador,
                "tipo": tipo
            })
            
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
# 3. INTERFAZ VISUAL FINAL
# ==========================================
st.title("🏆 Bot Predictor Pro - Especial para la Quiniela del Trabajo")
st.write("Datos en tiempo real procesados matemáticamente para darte una conclusión lógica directa para tu concurso.")
st.markdown("---")

if df_partidos_real.empty:
    st.warning("Cargando la programación oficial y sincronizando horarios...")
else:
    st.sidebar.header("Filtros")
    grupos = ["Todos"] + sorted(list(df_partidos_real["Grupo"].unique()))
    grupo_sel = st.sidebar.selectbox("Selecciona un Grupo:", grupos)
    
    df_filtrado = df_partidos_real if grupo_sel == "Todos" else df_partidos_real[df_partidos_real["Grupo"] == grupo_sel]
    
    for index, fila in df_filtrado.iterrows():
        local = fila["Local"]
        visitante = fila["Visitante"]
        grupo = fila["Grupo"]
        fecha = fila["Fecha"]
        hora = fila["Hora"]
        
        res = calcular_prediccion_concurso(local, visitante)
        g_l, g_v = res["Marcador_Concurso"]
        
        if res["Tendencia"] == "LOCAL": texto_conclusion = f"🎯 GANADOR: {local}"
        elif res["Tendencia"] == "VISITANTE": texto_conclusion = f"🎯 GANADOR: {visitante}"
        else: texto_conclusion = "🎯 RESULTADO: Empate"
        
        # El título despliega la hora real exacta ajustada a Lima
        with st.expander(f"📅 {fecha} a las {hora} hs (Hora Perú) | {grupo}: {local} vs {visitante}"):
            
            st.info(f"### 📋 RECOMENDACIÓN PARA TU QUINIELA:\n**{texto_conclusion}** con un marcador exacto de **{local} {g_l} - {g_v} {visitante}**")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**📊 Respaldo de Probabilidades Generales:**")
                st.write(f"Victoria {local}: {res['P_Local']}%")
                st.progress(int(res['P_Local']))
                st.write(f"Empate: {res['P_Empate']}%")
                st.progress(int(res['P_Empate']))
                st.write(f"Victoria {visitante}: {res['P_Visitante']}%")
                st.progress(int(res['P_Visitante']))
                
            with col2:
                st.write("**🎲 Marcadores alternativos sueltos:**")
                for _, m_fila in res["Top_Marcadores"].iterrows():
                    ml, mv = m_fila["marcador"]
                    porc = round(m_fila["prob"] * 100, 1)
                    st.write(f"• {local} {ml} - {mv} {visitante} ({porc}%)")
