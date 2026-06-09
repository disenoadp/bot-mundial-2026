import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from difflib import get_close_matches
import io

# Configuración
st.set_page_config(page_title="Bot Predictor 2026", layout="wide")
API_TOKEN = st.secrets.get("FOOTBALL_API_TOKEN", "")
BASE_URL = "https://api.football-data.org/v4/"

# URL ALTERNATIVA MÁS ESTABLE
URL_HISTORICO = "https://raw.githubusercontent.com/martivo/football-data-analysis/master/results.csv"

# ==========================================
# 1. CARGA CON MANEJO DE ERRORES ROBUSTO
# ==========================================
@st.cache_data(ttl=3600)
def cargar_todo():
    # A) API
    headers = {"X-Auth-Token": API_TOKEN} if API_TOKEN else {}
    resp = requests.get(f"{BASE_URL}competitions/WC/matches", headers=headers)
    
    if resp.status_code != 200:
        st.error(f"Error API: {resp.status_code}. Revisa tu TOKEN.")
        return {}, {}, []
        
    partidos_api = resp.json().get("matches", [])
    equipos_api = set()
    for m in partidos_api:
        if m.get("homeTeam"): equipos_api.add(m["homeTeam"]["name"])
        if m.get("awayTeam"): equipos_api.add(m["awayTeam"]["name"])
    
    # B) CSV con manejo de errores
    try:
        s = requests.get(URL_HISTORICO).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
    except Exception as e:
        st.error(f"No se pudo descargar el histórico: {e}")
        return {}, {}, partidos_api

    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= 2015]
    equipos_csv = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    
    # C) Mapa y Stats
    mapa = {eq: get_close_matches(eq, equipos_csv, n=1, cutoff=0.6)[0] 
            for eq in equipos_api if get_close_matches(eq, equipos_csv, n=1, cutoff=0.6)}
            
    stats = {}
    for eq in equipos_csv:
        sub = df[(df['home_team'] == eq) | (df['away_team'] == eq)]
        if len(sub) > 0:
            goles = sub.apply(lambda x: x['home_score'] if x['home_team'] == eq else x['away_score'], axis=1).sum()
            stats[eq] = {'g': goles, 'p': len(sub)}
            
    return mapa, stats, partidos_api

MAPA, STATS, PARTIDOS = cargar_todo()

# ==========================================
# 2. LÓGICA DE PREDICCIÓN Y UI
# ==========================================
def predecir(local, visitante):
    l_csv = MAPA.get(local, local)
    v_csv = MAPA.get(visitante, visitante)
    s_l = STATS.get(l_csv, {'g': 1.5, 'p': 1})
    s_v = STATS.get(v_csv, {'g': 1.5, 'p': 1})
    
    fuerza_l = s_l['g'] / s_l['p']
    fuerza_v = s_v['g'] / s_v['p']
    return poisson.pmf(np.arange(6), fuerza_l), poisson.pmf(np.arange(6), fuerza_v), s_l, s_v

st.title("⚽ Bot Predictor Autogestionado")

if not PARTIDOS:
    st.warning("La API no devolvió partidos. Verifica que el torneo esté activo.")
else:
    for m in PARTIDOS:
        l = m["homeTeam"]["name"]
        v = m["awayTeam"]["name"]
        p_l, p_v, s_l, s_v = predecir(l, v)
        
        with st.expander(f"{l} vs {v}"):
            st.write(f"Historial: {l}({s_l['p']} part) vs {v}({s_v['p']} part)")
            matrix = np.outer(p_l, p_v)
            c1, c2, c3 = st.columns(3)
            c1.metric("Gana Local", f"{np.sum(np.tril(matrix, -1)):.1%}")
            c2.metric("Empate", f"{np.sum(np.diag(matrix)):.1%}")
            c3.metric("Gana Visitante", f"{np.sum(np.triu(matrix, 1)):.1%}")
