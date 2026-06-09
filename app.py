import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from difflib import get_close_matches
import io

st.set_page_config(page_title="Bot Predictor 2026", layout="wide")
API_TOKEN = st.secrets.get("FOOTBALL_API_TOKEN", "")
BASE_URL = "https://api.football-data.org/v4/"
URL_HISTORICO = "https://raw.githubusercontent.com/martivo/football-data-analysis/master/results.csv"

@st.cache_data(ttl=3600)
def cargar_todo():
    # 1. Obtener Partidos API
    headers = {"X-Auth-Token": API_TOKEN} if API_TOKEN else {}
    resp = requests.get(f"{BASE_URL}competitions/WC/matches", headers=headers)
    partidos_api = resp.json().get("matches", []) if resp.status_code == 200 else []
    equipos_api = {m["homeTeam"]["name"] for m in partidos_api if "homeTeam" in m}.union(
                  {m["awayTeam"]["name"] for m in partidos_api if "awayTeam" in m})

    # 2. Cargar CSV con detección automática
    try:
        s = requests.get(URL_HISTORICO).content
        df = pd.read_csv(io.StringIO(s.decode('utf-8')))
        
        # Detectar columnas dinámicamente
        cols = [c.lower() for c in df.columns]
        date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
        home_col = next((c for c in df.columns if 'home' in c.lower()), df.columns[1])
        away_col = next((c for c in df.columns if 'away' in c.lower()), df.columns[2])
        
        df = df.rename(columns={date_col: 'date', home_col: 'home_team', away_col: 'away_team'})
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'].dt.year >= 2015]
    except Exception as e:
        st.error(f"Error procesando el CSV: {e}")
        return {}, {}, partidos_api

    # 3. Mapeo automático
    equipos_csv = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    mapa = {}
    for eq in equipos_api:
        match = get_close_matches(eq, equipos_csv, n=1, cutoff=0.6)
        if match: mapa[eq] = match[0]
            
    # 4. Stats
    stats = {}
    for eq in equipos_csv:
        sub = df[(df['home_team'] == eq) | (df['away_team'] == eq)]
        if not sub.empty:
            # Detectar columna de goles
            g_col = next((c for c in sub.columns if 'score' in c.lower()), None)
            goles = sub.apply(lambda x: x['home_score'] if x['home_team'] == eq else x['away_score'], axis=1).sum()
            stats[eq] = {'g': goles, 'p': len(sub)}
            
    return mapa, stats, partidos_api

MAPA, STATS, PARTIDOS = cargar_todo()

# UI (La lógica de predicción se mantiene igual)
st.title("⚽ Bot Predictor Autogestionado")
if not PARTIDOS:
    st.warning("No hay partidos activos en la API.")
else:
    for m in PARTIDOS:
        l, v = m["homeTeam"]["name"], m["awayTeam"]["name"]
        l_csv, v_csv = MAPA.get(l, l), MAPA.get(v, v)
        s_l = STATS.get(l_csv, {'g': 1.5, 'p': 1})
        s_v = STATS.get(v_csv, {'g': 1.5, 'p': 1})
        
        fuerza_l = s_l['g'] / s_l['p']
        fuerza_v = s_v['g'] / s_v['p']
        p_l = poisson.pmf(np.arange(6), fuerza_l)
        p_v = poisson.pmf(np.arange(6), fuerza_v)
        
        with st.expander(f"{l} vs {v}"):
            matrix = np.outer(p_l, p_v)
            c1, c2, c3 = st.columns(3)
            c1.metric("Local", f"{np.sum(np.tril(matrix, -1)):.1%}")
            c2.metric("Empate", f"{np.sum(np.diag(matrix)):.1%}")
            c3.metric("Visitante", f"{np.sum(np.triu(matrix, 1)):.1%}")
