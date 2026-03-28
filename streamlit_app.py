import time
import streamlit as st
import pandas as pd
import numpy as np
import requests
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.express as px
from itertools import combinations

st.set_page_config(page_title="Pair Trading", layout="wide")

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-size: 13px !important; }
h1 { font-size: 17px !important; font-weight: 500 !important; }
h2, h3 { font-size: 13px !important; font-weight: 500 !important; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 1rem !important; }
[data-testid="stSidebar"] { background-color: #fafaf8 !important; min-width: 200px !important; max-width: 200px !important; }
[data-testid="stSidebar"] label { font-size: 11px !important; color: #888 !important; }
.stButton > button { background: #1a1a1a !important; color: #fff !important; border: none !important; border-radius: 6px !important; font-size: 12px !important; height: 32px !important; }
.stButton > button:hover { background: #333 !important; }
[data-testid="metric-container"] { background: #f7f6f3 !important; border-radius: 8px !important; padding: 10px 14px !important; border: none !important; }
[data-testid="stMetricLabel"] { font-size: 10px !important; color: #999 !important; }
[data-testid="stMetricValue"] { font-size: 20px !important; font-weight: 500 !important; }
[data-testid="stSelectbox"] label, [data-testid="stNumberInput"] label { font-size: 11px !important; color: #888 !important; }
[data-testid="stAlert"] { font-size: 12px !important; padding: 8px 14px !important; }
button[data-baseweb="tab"] { font-size: 12px !important; }
.stApp { background-color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY = "CG-zQg6pyzA4RPm5Tti2p7RTsn2"

CRYPTOS = {
    "Bitcoin":     "bitcoin",
    "Ethereum":    "ethereum",
    "Aave":        "aave",
    "Pendle":      "pendle",
    "Morpho":      "morpho",
    "Pump.fun":    "pump-fun",
    "LayerZero":   "layerzero",
    "Hyperliquid": "hyperliquid",
    "Syrup":       "maple-finance",
    "Fluid":       "fluid",
}

METRICS_INFO = {
    "Corrélation": {
        "emoji": "📊",
        "definition": "Mesure si deux actifs bougent dans le même sens. "
                      "Proche de **1.0** = ils bougent ensemble (idéal pour le pair trading). "
                      "Proche de **0** = ils sont indépendants. Négatif = ils bougent en sens inverse.",
        "seuil": "✅ Bon signal si > 0.7",
    },
    "Hedge Ratio (β)": {
        "emoji": "⚖️",
        "definition": "Le réglage de l'équilibre entre les deux actifs. "
                      "Indique combien d'unités de l'actif B il faut pour couvrir 1 unité de l'actif A. "
                      "Permet de construire une position **market-neutral**.",
        "seuil": "ℹ️ Pas de seuil fixe — dépend de la paire",
    },
    "Co-intégration (p)": {
        "emoji": "🔬",
        "definition": "La preuve mathématique qu'un **élastique invisible** relie les deux prix. "
                      "Si p < 0.05, l'écart entre les deux actifs revient toujours vers sa moyenne. "
                      "C'est le fondement théorique du pair trading.",
        "seuil": "✅ Bon signal si p < 0.05",
    },
    "Half-Life (jours)": {
        "emoji": "⏳",
        "definition": "Le chrono du trade. Temps moyen estimé pour que l'écart (spread) "
                      "se réduise de moitié et revienne vers l'équilibre. "
                      "Trop court = bruit. Trop long = capital immobilisé trop longtemps.",
        "seuil": "✅ Idéal entre 5 et 15 jours",
    },
    "Z-Score": {
        "emoji": "🌡️",
        "definition": "Le thermomètre de l'opportunité. Mesure à quel point l'écart actuel "
                      "est éloigné de sa moyenne historique (en nombre d'écarts-types). "
                      "**+2** = actif A trop cher vs B → signal SHORT A / LONG B. "
                      "**-2** = actif A trop bon marché vs B → signal LONG A / SHORT B.",
        "seuil": "🚨 Signal fort si |z| > 2",
    },
}

# ─── FONCTIONS ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(coin_id, days=180):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}
    headers = {"x-cg-demo-api-key": API_KEY}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code == 429:
            time.sleep(12)
            r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200:
            return None, f"Erreur API {r.status_code} pour {coin_id}"
        prices = r.json().get("prices", [])
        if len(prices) < 40:
            return None, f"{coin_id} : historique insuffisant ({len(prices)} jours disponibles)"
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df = df.drop_duplicates("date").set_index("date")["price"]
        return df, None
    except Exception as e:
        return None, str(e)


def compute_metrics(series_a, series_b, name_a, name_b):
    df = pd.concat([series_a, series_b], axis=1).dropna()
    df.columns = ["A", "B"]
    if len(df) < 40:
        return None

    returns = df.pct_change().dropna()
    correlation = returns["A"].corr(returns["B"])

    X = sm.add_constant(df["B"])
    model = sm.OLS(df["A"], X).fit()
    beta = model.params["B"]
    alpha = model.params["const"]

    spread = df["A"] - (beta * df["B"] + alpha)
    try:
        p_value = adfuller(spread.dropna())[1]
    except Exception:
        p_value = 1.0

    z_score = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
    current_z = float(z_score.iloc[-1])

    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    valid = ~(spread_lag.isna() | spread_diff.isna())
    try:
        res = sm.OLS(spread_diff[valid], spread_lag[valid]).fit()
        lambda_val = -res.params.iloc[0]
        half_life = np.log(2) / lambda_val if lambda_val > 0 else float("inf")
    except Exception:
        half_life = float("inf")

    if p_value < 0.05 and half_life < 15:
        verdict = "✅ Idéale"
        verdict_color = "green"
    elif p_value < 0.05:
        verdict = "⚠️ Lente"
        verdict_color = "orange"
    else:
        verdict = "❌ Faible"
        verdict_color = "red"

    if current_z > 2:
        signal = f"SHORT {name_a} / LONG {name_b}"
    elif current_z < -2:
        signal = f"LONG {name_a} / SHORT {name_b}"
    else:
        signal = "Pas de signal"

    return {
        "Corrélation": round(correlation, 3),
        "Hedge Ratio (β)": round(float(beta), 4),
        "Co-intégration (p)": round(p_value, 4),
        "Half-Life (jours)": round(half_life, 1) if half_life != float("inf") else "∞",
        "Z-Score": round(current_z, 2),
        "Verdict": verdict,
        "Signal": signal,
        "spread": spread,
        "z_score": z_score,
        "df": df,
        "verdict_color": verdict_color,
    }

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Pair Trading Analyzer")
st.caption("Données : CoinGecko API · 6 mois · quotidien")

tabs = st.tabs(["📚 Les 5 métriques", "🔍 Analyse d'une paire", "🗺️ Matrice des paires"])

# ══ TAB 1 — MÉTRIQUES ══════════════════════════════════════════════════════════
with tabs[0]:
    st.subheader("Les 5 métriques du pair trading")
    st.markdown("Maîtrise-les dans l'ordre : chacune s'appuie sur la précédente.")

    for i, (name, info) in enumerate(METRICS_INFO.items(), 1):
        with st.expander(f"{info['emoji']}  {i}. {name}", expanded=True):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(info["definition"])
            with col2:
                st.info(info["seuil"])

    st.divider()
    st.markdown("### Comment lire les résultats ensemble ?")
    st.markdown(
        "| Étape | Métrique | Question posée |\n"
        "|-------|----------|----------------|\n"
        "| 1 | **Corrélation** | Est-ce que les deux actifs bougent ensemble ? |\n"
        "| 2 | **Hedge Ratio** | Dans quelle proportion les positionner ? |\n"
        "| 3 | **Co-intégration** | Y a-t-il un élastique entre eux ? |\n"
        "| 4 | **Half-Life** | Combien de temps dure un trade typique ? |\n"
        "| 5 | **Z-Score** | L'élastique est-il tendu *maintenant* ? |\n"
    )

# ══ TAB 2 — ANALYSE D'UNE PAIRE ════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Analyse d'une paire")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        name_a = st.selectbox("Actif A", list(CRYPTOS.keys()), index=2)
    with col2:
        name_b = st.selectbox("Actif B", list(CRYPTOS.keys()), index=3)
    with col3:
        capital = st.number_input("Capital ($)", value=1000, step=100)

    if name_a == name_b:
        st.warning("Choisis deux actifs différents.")
    elif st.button("🚀 Analyser", use_container_width=True):
        with st.spinner(f"Chargement {name_a}..."):
            s_a, err_a = fetch_prices(CRYPTOS[name_a])
        with st.spinner(f"Chargement {name_b}..."):
            s_b, err_b = fetch_prices(CRYPTOS[name_b])

        if err_a:
            st.error(f"❌ {name_a} : {err_a}")
        elif err_b:
            st.error(f"❌ {name_b} : {err_b}")
        else:
            m = compute_metrics(s_a, s_b, name_a, name_b)
            if m is None:
                st.error("Pas assez de données communes pour calculer.")
            else:
                # Verdict
                if m["verdict_color"] == "green":
                    st.success(f"**{m['Verdict']}** — co-intégration solide, half-life rapide.")
                elif m["verdict_color"] == "orange":
                    st.warning(f"**{m['Verdict']}** — co-intégration ok mais paire lente.")
                else:
                    st.error(f"**{m['Verdict']}** — co-intégration insuffisante.")

                # Signal + allocation
                z = m["Z-Score"]
                beta = m["Hedge Ratio (β)"]
                p_a = float(s_a.iloc[-1])
                p_b = float(s_b.iloc[-1])
                ratio = abs(beta * p_b / p_a)
                alloc_a = capital / (1 + ratio)
                alloc_b = capital - alloc_a

                if abs(z) > 2:
                    st.error(
                        f"🚨 **Signal : {m['Signal']}**\n\n"
                        f"→ {name_a} : **{alloc_a:.0f}$**  ·  {name_b} : **{alloc_b:.0f}$**"
                    )
                else:
                    st.info(f"😴 **{m['Signal']}** — z-score neutre ({z})")

                # Métriques
                st.divider()
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Corrélation", m["Corrélation"])
                c2.metric("Hedge Ratio β", m["Hedge Ratio (β)"])
                c3.metric("Co-intégration p", m["Co-intégration (p)"])
                c4.metric("Half-Life", f"{m['Half-Life (jours)']} j")
                c5.metric("Z-Score", m["Z-Score"])

                # Graphiques
                df = m["df"]
                z_score = m["z_score"]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["A"] / df["A"].iloc[0],
                    name=name_a, line=dict(color="#1D9E75", width=1.5)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index, y=df["B"] / df["B"].iloc[0],
                    name=name_b, line=dict(color="#7F77DD", width=1.5)
                ))
                fig.update_layout(
                    title=dict(text="Prix normalisés (base 1)", font=dict(size=12)),
                    height=240, margin=dict(t=36, b=16, l=40, r=16),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                    legend=dict(font=dict(size=11)),
                )
                fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
                fig.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=z_score.index, y=z_score,
                    name="Z-Score", line=dict(color="#378ADD", width=1.5),
                    fill="tozeroy", fillcolor="rgba(55,138,221,0.05)"
                ))
                for y_val, color in [(2, "rgba(220,50,50,0.5)"), (-2, "rgba(220,50,50,0.5)"), (0, "rgba(180,180,180,0.5)")]:
                    fig2.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1)
                fig2.update_layout(
                    title=dict(text="Z-Score — signal de trading", font=dict(size=12)),
                    height=240, margin=dict(t=36, b=16, l=40, r=16),
                    plot_bgcolor="#fff", paper_bgcolor="#fff",
                )
                fig2.update_xaxes(showgrid=False, tickfont=dict(size=10))
                fig2.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
                st.plotly_chart(fig2, use_container_width=True)

# ══ TAB 3 — MATRICE ════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("Matrice des paires")
    st.caption("Toutes les combinaisons possibles entre les cryptos sélectionnées.")

    selected = st.multiselect(
        "Cryptos à inclure",
        list(CRYPTOS.keys()),
        default=list(CRYPTOS.keys()),
    )

    if len(selected) < 2:
        st.warning("Sélectionne au moins 2 cryptos.")
    elif st.button("🗺️ Calculer toutes les paires", use_container_width=True):

        pairs = list(combinations(selected, 2))
        progress = st.progress(0, text="Chargement des données...")

        # Téléchargement avec délai pour respecter le rate limit
        cache = {}
        errors = []
        for i, name in enumerate(selected):
            cache[name], err = fetch_prices(CRYPTOS[name])
            if err:
                errors.append(f"{name} : {err}")
            time.sleep(1)
            progress.progress((i + 1) / len(selected), text=f"Chargement {name}...")

        if errors:
            for e in errors:
                st.warning(f"⚠️ {e}")

        # Calcul des paires
        results = []
        for i, (a, b) in enumerate(pairs):
            progress.progress((i + 1) / len(pairs), text=f"Calcul {a} / {b}...")
            if cache.get(a) is None or cache.get(b) is None:
                continue
            m = compute_metrics(cache[a], cache[b], a, b)
            if m is None:
                continue
            results.append({
                "Paire": f"{a} / {b}",
                "Corrélation": m["Corrélation"],
                "p-value": m["Co-intégration (p)"],
                "Half-Life (j)": m["Half-Life (jours)"],
                "Z-Score": m["Z-Score"],
                "Verdict": m["Verdict"],
                "Signal": m["Signal"],
            })

        progress.empty()

        if not results:
            st.error("Aucun résultat. Vérifie ta connexion.")
        else:
            df_res = pd.DataFrame(results)

            # Matrice de corrélation
            st.markdown("#### Corrélations")
            corr_matrix = pd.DataFrame(index=selected, columns=selected, dtype=float)
            for _, row in df_res.iterrows():
                a, b = row["Paire"].split(" / ")
                corr_matrix.loc[a, b] = row["Corrélation"]
                corr_matrix.loc[b, a] = row["Corrélation"]
            for name in selected:
                corr_matrix.loc[name, name] = 1.0

            fig_corr = px.imshow(
                corr_matrix.astype(float),
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1,
                text_auto=".2f",
                aspect="auto",
                height=420,
            )
            fig_corr.update_layout(
                paper_bgcolor="#fff", plot_bgcolor="#fff",
                margin=dict(t=16, b=16),
                font=dict(size=11),
            )
            fig_corr.update_traces(textfont_size=11)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Tableau
            st.markdown("#### Toutes les paires")

            def color_verdict(val):
                if "✅" in str(val): return "background-color: rgba(29,158,117,0.15)"
                if "⚠️" in str(val): return "background-color: rgba(239,159,39,0.15)"
                return "background-color: rgba(226,75,74,0.1)"

            def color_pvalue(val):
                try: return "color: #1D9E75" if float(val) < 0.05 else "color: #E24B4A"
                except: return ""

            def color_zscore(val):
                try: return "font-weight: 500; color: #E24B4A" if abs(float(val)) > 2 else ""
                except: return ""

            styled = (
                df_res.style
                .applymap(color_verdict, subset=["Verdict"])
                .applymap(color_pvalue, subset=["p-value"])
                .applymap(color_zscore, subset=["Z-Score"])
                .format({"Corrélation": "{:.3f}", "p-value": "{:.4f}", "Z-Score": "{:.2f}"})
            )
            st.dataframe(styled, use_container_width=True, height=480)

            # Top 5
            st.markdown("#### 🏆 Meilleures configurations")
            ideal = df_res[df_res["Verdict"] == "✅ Idéale"].sort_values("p-value")
            if len(ideal) == 0:
                st.info("Aucune paire idéale sur cette période. Consulte le tableau.")
            else:
                for _, row in ideal.head(5).iterrows():
                    st.success(
                        f"**{row['Paire']}** · "
                        f"corr {row['Corrélation']} · "
                        f"p {row['p-value']} · "
                        f"hl {row['Half-Life (j)']} j · "
                        f"z {row['Z-Score']} · "
                        f"{row['Signal']}"
                    )
