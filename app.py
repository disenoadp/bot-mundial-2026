import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from difflib import get_close_matches

# Configuración inicial
st.set_page_config(page_title="Bot Predictor Inteligente", layout="wide")
API_TOKEN = st.secrets["FOOTBALL_API_TOKEN"]
BASE_URL = "https://api.football-data.org/v4/"
URL_HISTORICO = "https://raw.githubusercontent.com/datasets/football-data/master/data/international-results.csv"

# ==========================================
# 1. MOTOR DE AUTO-MAPEO Y PROCESAMIENTO
# ==========================================
@st.cache_data(ttl=3600)
def cargar_todo():
    # A) Descargar partidos de la API
    resp = requests.get(f"{BASE_URL}competitions/WC/matches", headers={"X-Auth-Token": API_TOKEN})
    partidos_api = resp.json().get("matches", [])
    equipos_api = set()
    for m in partidos_api:
        if m.get("homeTeam"): equipos_api.add(m["homeTeam"]["name"])
        if m.get("awayTeam"): equipos_api.add(m["awayTeam"]["name"])
    
    # B) Cargar CSV y extraer equipos únicos
    df = pd.read_csv(URL_HISTORICO)
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2015] # Solo era moderna
    equipos_csv = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    
    # C) Generar mapa automático usando similitud de cadenas
    mapa = {}
    for eq in equipos_api:
        match = get_close_matches(eq, equipos_csv, n=1, cutoff=0.6)
        if match: mapa[eq] = match[0]
            
    # D) Calcular estadísticas históricas reales
    stats = {}
    for eq in equipos_csv:
        sub = df[(df['home_team'] == eq) | (df['away_team'] == eq)]
        pj = len(sub)
        if pj > 0:
            goles = sub.apply(lambda x: x['home_score'] if x['home_team'] == eq else x['away_score'], axis=1).sum()
            stats[eq] = {'g': goles, 'p': pj}
            
    return mapa, stats, partidos_api

MAPA, STATS, PARTIDOS = cargar_todo()

# ==========================================
# 2. LÓGICA DE PREDICCIÓN
# ==========================================
def predecir(local, visitante):
    # Obtener nombres reales en CSV
    l_csv = MAPA.get(local, local)
    v_csv = MAPA.get(visitante, visitante)
    
    # Obtener stats reales
    s_l = STATS.get(l_csv, {'g': 1.5, 'p': 1})
    s_v = STATS.get(v_csv, {'g': 1.5, 'p': 1})
    
    # Poisson
    fuerza_l = s_l['g'] / s_l['p']
    fuerza_v = s_v['g'] / s_v['p']
    
    return poisson.pmf(np.arange(6), fuerza_l), poisson.pmf(np.arange(6), fuerza_v), s_l, s_v

# ==========================================
# 3. INTERFAZ
# ==========================================
st.title("⚽ Bot Predictor Autogestionado")
for m in PARTIDOS:
    l = m["homeTeam"]["name"]
    v = m["awayTeam"]["name"]
    p_l, p_v, s_l, s_v = predecir(l, v)
    
    with st.expander(f"{l} vs {v}"):
        st.write(f"Partidos analizados en histórico: {l}({s_l['p']}) vs {v}({s_v['p']})")
        # Matriz de probabilidad simple
        matrix = np.outer(p_l, p_v)
        local_win = np.sum(np.tril(matrix, -1))
        draw = np.sum(np.diag(matrix))
        away_win = np.sum(np.triu(matrix, 1))
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Gana Local", f"{local_win:.1%}")
        c2.metric("Empate", f"{draw:.1%}")
        c3.metric("Gana Visitante", f"{away_win:.1%}")
