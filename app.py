import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
from difflib import get_close_matches

st.set_page_config(page_title="Bot Predictor 2026", layout="wide")

# Configuración
API_TOKEN = st.secrets.get("FOOTBALL_API_TOKEN", "")
BASE_URL = "https://api.football-data.org/v4/"

@st.cache_data
def cargar_datos_locales():
    # 1. Cargar archivo local (asegúrate de que 'resultados.csv' esté en la carpeta del proyecto)
    try:
        df = pd.read_csv('resultados.csv')
    except FileNotFoundError:
        st.error("Error: El archivo 'resultados.csv' no se encuentra en el repositorio.")
        return {}, {}, []

    # Estandarizar nombres de columnas (ajusta esto si tu CSV tiene nombres distintos)
    df.columns = [c.lower() for c in df.columns]
    
    # 2. Obtener partidos de API
    headers = {"X-Auth-Token": API_TOKEN} if API_TOKEN else {}
    resp = requests.get(f"{BASE_URL}competitions/WC/matches", headers=headers)
    partidos = resp.json().get("matches", []) if resp.status_code == 200 else []
    
    # 3. Procesar stats reales
    stats = {}
    equipos_csv = set(df['home_team'].unique()).union(set(df['away_team'].unique()))
    
    for eq in equipos_csv:
        sub = df[(df['home_team'] == eq) | (df['away_team'] == eq)]
        if len(sub) > 0:
            goles = sub.apply(lambda x: x['home_score'] if x['home_team'] == eq else x['away_score'], axis=1).sum()
            stats[eq] = {'g': goles, 'p': len(sub)}
            
    return stats, partidos

STATS, PARTIDOS = cargar_datos_locales()

# Lógica de emparejamiento con el CSV local
def get_stats(nombre_api):
    # Intentar buscar coincidencia cercana en los equipos del CSV
    nombres_csv = list(STATS.keys())
    match = get_close_matches(nombre_api, nombres_csv, n=1, cutoff=0.6)
    if match:
        return STATS[match[0]], match[0]
    return {'g': 1.5, 'p': 1}, None # Valor neutro

st.title("⚽ Bot Predictor Estable")

for m in PARTIDOS:
    l, v = m["homeTeam"]["name"], m["awayTeam"]["name"]
    s_l, n_l = get_stats(l)
    s_v, n_v = get_stats(v)
    
    with st.expander(f"{l} vs {v}"):
        st.write(f"Comparando: {l} ({n_l}) vs {v} ({n_v})")
        # Poisson y visualización (igual que antes)
        f_l, f_v = s_l['g']/s_l['p'], s_v['g']/s_v['p']
        # ... (aquí iría tu lógica de cálculo de probabilidades)
