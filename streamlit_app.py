import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- CONFIG PAGE ---
st.set_page_config(page_title="Pair Trading Analyzer", layout="wide")
st.title("📊 Analyse de Paire Statistique")

# --- INPUTS DANS LA SIDEBAR ---
st.sidebar.header("Paramètres")
ticker_a = st.sidebar.text_input("Actif A", value="AAVE-USD")
ticker_b = st.sidebar.text_input("Actif B", value="PENDLE-USD")
capital_total = st.sidebar.number_input("Capital ($)", value=1000, step=100)
period = st.sidebar.selectbox("Période", ["3mo", "6mo", "1y"], index=1)

run = st.sidebar.button("🚀 Analyser", use_container_width=True)

# --- ANALYSE ---
if run:
    with st.spinner("Téléchargement des données..."):
        data_a = yf.download(ticker_a, period=period, interval="1d", progress=False)['Close']
        data_b = yf.download(ticker_b, period=period, interval="1d", progress=False)['Close']
        df = pd.concat([data_a, data_b], axis=1).dropna()
        df.columns = ['Price_A', 'Price_B']

    # --- CALCULS (identiques à ton code original) ---
    returns = df.pct_change().dropna()
    correlation = returns['Price_A'].corr(returns['Price_B'])

    X = sm.add_constant(df['Price_B'])
    model = sm.OLS(df['Price_A'], X).fit()
    beta = model.params['Price_B']
    alpha = model.params['const']

    spread = df['Price_A'] - (beta * df['Price_B'] + alpha)
    p_value = adfuller(spread.dropna())[1]

    z_score = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    valid = ~(spread_lag.isna() | spread_diff.isna())
    res = sm.OLS(spread_diff[valid], spread_lag[valid]).fit()
    lambda_val = -res.params.iloc[0]
    half_life = np.log(2) / lambda_val if lambda_val > 0 else float('inf')

    actual_z = z_score.iloc[-1]
    price_a_now = df['Price_A'].iloc[-1]
    price_b_now = df['Price_B'].iloc[-1]

    ratio_capital = abs(beta * price_b_now / price_a_now)
    alloc_a = capital_total / (1 + ratio_capital)
    alloc_b = capital_total - alloc_a

    # --- VERDICT & SIGNAL ---
    ideal = p_value < 0.05 and half_life < 15

    if actual_z > 2:
        signal = f"🚨 SHORT {ticker_a} / LONG {ticker_b}"
        signal_detail = f"Vendre **{alloc_a:.2f}$** de {ticker_a} · Acheter **{alloc_b:.2f}$** de {ticker_b}"
        signal_color = "inverse"
    elif actual_z < -2:
        signal = f"🚨 LONG {ticker_a} / SHORT {ticker_b}"
        signal_detail = f"Acheter **{alloc_a:.2f}$** de {ticker_a} · Vendre **{alloc_b:.2f}$** de {ticker_b}"
        signal_color = "inverse"
    else:
        signal = "😴 Pas de signal — marché à l'équilibre"
        signal_detail = "Le z-score est entre -2 et +2, aucune opportunité statistique."
        signal_color = "off"

    # --- AFFICHAGE ---

    # Verdict bannière
    if ideal:
        st.success("✅ Configuration IDÉALE — co-intégration solide et half-life rapide.")
    else:
        st.warning("⚠️ Prudence — co-intégration faible ou paire trop lente.")

    st.info(f"**{signal}**\n\n{signal_detail}")

    # Métriques en colonnes
    st.subheader("Statistiques clés")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Corrélation", f"{correlation:.2f}", help="Proche de 1 = paire très liée")
    col2.metric("Hedge Ratio (β)", f"{beta:.4f}", help="Unités de B pour couvrir 1 unité de A")
    col3.metric("Co-intégration (p)", f"{p_value:.4f}", delta="✅ < 0.05" if p_value < 0.05 else "❌ > 0.05", delta_color="normal" if p_value < 0.05 else "inverse")
    col4.metric("Half-Life", f"{half_life:.1f} j", help="Temps moyen de retour à la moyenne")

    # Z-score actuel bien visible
    st.metric("Z-Score actuel", f"{actual_z:.2f}", help="Signal fort si > +2 ou < -2")

    # --- GRAPHIQUES ---
    st.subheader("Graphiques")

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            f"Prix normalisés — {ticker_a} vs {ticker_b}",
            "Spread",
            "Z-Score (signal de trading)"
        ),
        vertical_spacing=0.08,
        row_heights=[0.35, 0.3, 0.35]
    )

    # Prix normalisés
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Price_A'] / df['Price_A'].iloc[0],
        name=ticker_a, line=dict(color="#5DCAA5")
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=df.index, y=df['Price_B'] / df['Price_B'].iloc[0],
        name=ticker_b, line=dict(color="#7F77DD")
    ), row=1, col=1)

    # Spread
    fig.add_trace(go.Scatter(
        x=spread.index, y=spread,
        name="Spread", line=dict(color="#EF9F27"), fill='tozeroy', fillcolor='rgba(239,159,39,0.08)'
    ), row=2, col=1)

    # Z-score
    fig.add_trace(go.Scatter(
        x=z_score.index, y=z_score,
        name="Z-Score", line=dict(color="#378ADD")
    ), row=3, col=1)

    # Lignes de signal ±2 et 0
    for y_val, color, dash in [(2, "red", "dash"), (-2, "red", "dash"), (0, "gray", "dot")]:
        fig.add_hline(y=y_val, line_dash=dash, line_color=color, opacity=0.5, row=3, col=1)

    fig.update_layout(
        height=650,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(128,128,128,0.15)')

    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Configure tes paramètres dans la barre latérale et clique sur **Analyser**.")
