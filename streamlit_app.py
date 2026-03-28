import os
import glob
import streamlit as st
import pandas as pd
import numpy as np
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
DATA_DIR = "data"

def scan_tokens():
    files = glob.glob(os.path.join(DATA_DIR, "*-historical-data.csv"))
    tokens = {}
    for f in sorted(files):
        slug = os.path.basename(f).replace("-historical-data.csv", "")
        label = slug.replace("-", " ").title()
        tokens[label] = slug
    return tokens

CRYPTOS = scan_tokens()

if not CRYPTOS:
    st.error(f"Aucun fichier trouvé dans `{DATA_DIR}/`. Vérifie que tes CSV sont bien au format `nom-historical-data.csv`.")
    st.stop()

# ─── FONCTIONS ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(slug):
    path = os.path.join(DATA_DIR, f"{slug}-historical-data.csv")
    if not os.path.exists(path):
        return None, f"Fichier introuvable : {slug}-historical-data.csv"
    try:
        df = pd.read_csv(path)
        for col in df.columns:
            if col != "Date":
                df[col] = (
                    df[col].astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")
        if "Close" not in df.columns:
            return None, f"Colonne 'Close' introuvable dans {slug}-historical-data.csv"
        series = df["Close"].dropna()
        if len(series) < 40:
            return None, f"{slug} : seulement {len(series)} jours disponibles (minimum 40)"
        return series, None
    except Exception as e:
        return None, f"Erreur lecture {slug} : {str(e)}"


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

    ok_corr = correlation >= 0.7
    ok_p    = p_value < 0.05
    ok_hl   = half_life < 15
    ok_z    = abs(current_z) > 2

    if ok_corr and ok_p and ok_hl and ok_z:
        verdict = "✅ Valide"
        verdict_color = "green"
    else:
        verdict = "❌ Non valide"
        verdict_color = "red"

    if current_z > 2:
        signal = f"SHORT ↓ {name_a} / LONG ↑ {name_b}"
    elif current_z < -2:
        signal = f"LONG ↑ {name_a} / SHORT ↓ {name_b}"
    else:
        signal = "—"

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
st.caption(f"Données locales · {len(CRYPTOS)} tokens disponibles · dossier `data/`")

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "prefill_a" not in st.session_state:
    st.session_state.prefill_a = None
if "prefill_b" not in st.session_state:
    st.session_state.prefill_b = None
if "matrix_results" not in st.session_state:
    st.session_state["matrix_results"] = []

# ─── CALCUL AUTO AU DÉMARRAGE ──────────────────────────────────────────────────
_stale = any(
    "Idéale" in str(r.get("Verdict", "")) or
    "Pas de signal" not in [x.get("Verdict","") for x in st.session_state["matrix_results"]] and
    r.get("Corrélation", 1) < 0.7 and r.get("Verdict", "") == "✅ Valide"
    for r in st.session_state["matrix_results"]
)
if not st.session_state["matrix_results"] or _stale:
    all_names = list(CRYPTOS.keys())
    pairs = list(combinations(all_names, 2))
    bar = st.progress(0, text="Calcul des paires en cours...")
    price_cache = {}
    for name in all_names:
        price_cache[name], _ = fetch_prices(CRYPTOS[name])
    results_auto = []
    for i, (a, b) in enumerate(pairs):
        bar.progress((i + 1) / len(pairs), text=f"Calcul {a} / {b}…")
        if price_cache.get(a) is None or price_cache.get(b) is None:
            continue
        m = compute_metrics(price_cache[a], price_cache[b], a, b)
        if m is None:
            continue
        results_auto.append({
            "Paire":           f"{a} / {b}",
            "Corrélation":     m["Corrélation"],
            "Hedge Ratio β":   m["Hedge Ratio (β)"],
            "Co-intégration p": m["Co-intégration (p)"],
            "Half-Life":       m["Half-Life (jours)"],
            "Z-Score":         m["Z-Score"],
            "Verdict":         m["Verdict"],
            "Signal":          m["Signal"],
        })
    st.session_state["matrix_results"] = results_auto
    bar.empty()

# ══ MÉTRIQUES ══════════════════════════════════════════════════════════════════
METRICS_COMPACT = [
    {
        "emoji": "📊",
        "name": "Corrélation",
        "seuil": "> 0.7",
        "formule": "ρ = cov(r<sub>A</sub>, r<sub>B</sub>) / (σ<sub>A</sub> · σ<sub>B</sub>)",
        "note": "Calculée sur les rendements journaliers (pas les prix). Mesure si les deux actifs bougent dans le même sens.",
    },
    {
        "emoji": "⚖️",
        "name": "Hedge Ratio β",
        "seuil": "Pas de seuil",
        "formule": "β = cov(A, B) / var(B)",
        "note": "Régression OLS de A sur B. Indique combien d'unités de B couvrent 1 unité de A.",
    },
    {
        "emoji": "🔬",
        "name": "Co-intégration p",
        "seuil": "< 0.05",
        "formule": "ADF(A − βB − α) → p-value",
        "note": "Test de stationnarité du spread. Si p < 0.05, l'écart entre les deux prix revient toujours à sa moyenne.",
    },
    {
        "emoji": "⏳",
        "name": "Half-Life",
        "seuil": "5–15 jours",
        "formule": "t<sub>½</sub> = ln(2) / λ &nbsp;&nbsp; Δs<sub>t</sub> = λ · s<sub>t−1</sub>",
        "note": "Modèle Ornstein-Uhlenbeck. Temps moyen pour que l'écart se réduise de moitié.",
    },
    {
        "emoji": "🌡️",
        "name": "Z-Score",
        "seuil": "Signal si |z| > 2",
        "formule": "z = (s<sub>t</sub> − μ<sub>30</sub>) / σ<sub>30</sub>",
        "note": "Fenêtre glissante 30 jours. Un z > +2 se produit ~2.5% du temps — signal de trading.",
    },
]

cols = st.columns(5)
for col, info in zip(cols, METRICS_COMPACT):
    with col:
        st.markdown(
            f"""<div style="border:1px dashed #ccc;border-radius:8px;padding:14px 14px 14px;height:150px;display:flex;flex-direction:column;">
            <p style="font-size:12px;font-weight:500;margin:0 0 2px">{info['emoji']} {info['name']}</p>
            <p style="font-size:10px;color:#aaa;margin:0 0 8px">Seuil : {info['seuil']}</p>
            <p style="font-size:13px;font-family:Georgia,serif;text-align:center;margin:0 0 8px;color:#333">{info['formule']}</p>
            <p style="font-size:11px;color:#888;line-height:1.4;margin:0">{info['note']}</p>
            </div>""",
            unsafe_allow_html=True
        )

# ── Signaux actifs ────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Signaux actifs")
filtre = st.radio("", ["Valide uniquement", "Tout"], horizontal=True, label_visibility="collapsed")

if not st.session_state.get("matrix_results"):
    st.caption("Calcul en cours au prochain chargement…")
else:
    df_tab1 = pd.DataFrame(st.session_state["matrix_results"])

    if filtre == "Valide uniquement":
        df_tab1_signal = df_tab1[df_tab1["Verdict"] == "✅ Valide"].copy()
    else:
        df_tab1_signal = df_tab1.copy()

    df_tab1_signal = df_tab1_signal.sort_values("Verdict").reset_index(drop=True)

    if df_tab1_signal.empty:
        st.info("Aucun signal actif sur les paires calculées.")
    else:
        def _color_verdict(val):
            if "✅" in str(val): return "background-color:#e8f7f1;color:#0F6E56"
            return "background-color:#fdf0f0;color:#A32D2D"

        def _color_corr(val):
            try:
                v = float(val)
                if v >= 0.7: return "background-color:#e8f7f1;color:#0F6E56"
                if v >= 0.5: return "background-color:#fef3e2;color:#854F0B"
                return "background-color:#fdf0f0;color:#A32D2D"
            except: return ""

        def _color_p(val):
            try:
                v = float(val)
                if v < 0.05:  return "background-color:#e8f7f1;color:#0F6E56"
                if v < 0.15:  return "background-color:#fef3e2;color:#854F0B"
                return "background-color:#fdf0f0;color:#A32D2D"
            except: return ""

        def _color_hl(val):
            try:
                v = float(str(val).replace("∞", "9999"))
                if 5 <= v <= 15: return "background-color:#e8f7f1;color:#0F6E56"
                if v < 30:       return "background-color:#fef3e2;color:#854F0B"
                return "background-color:#fdf0f0;color:#A32D2D"
            except: return ""

        def _color_z(val):
            try:
                v = abs(float(val))
                if v > 2: return "background-color:#e8f7f1;color:#0F6E56;font-weight:500"
                return "background-color:#fdf0f0;color:#A32D2D"
            except: return ""

        def _color_signal(val):
            return ""  # pas de coloration sur le signal

        st.dataframe(
            df_tab1_signal.reset_index(drop=True).style
            .applymap(_color_verdict,  subset=["Verdict"])
            .applymap(_color_corr,     subset=["Corrélation"])
            .applymap(_color_p,        subset=["Co-intégration p"])
            .applymap(_color_hl,       subset=["Half-Life"])
            .applymap(_color_z,        subset=["Z-Score"])
            .applymap(_color_signal,   subset=["Signal"])
            .format({
                "Corrélation":      "{:.3f}",
                "Hedge Ratio β":    "{:.4f}",
                "Co-intégration p": "{:.4f}",
                "Z-Score":          "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Analyse de paire ─────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Analyse d'une paire")

keys = list(CRYPTOS.keys())
default_a = keys.index(st.session_state.prefill_a) if st.session_state.prefill_a in keys else 0
default_b = keys.index(st.session_state.prefill_b) if st.session_state.prefill_b in keys else min(1, len(keys) - 1)

ctrl1, ctrl2, ctrl3, ctrl4, _ = st.columns([1.2, 1.2, 0.8, 0.6, 1.2])
with ctrl1:
    name_a = st.selectbox("Actif A", keys, index=default_a, key="sel_a")
with ctrl2:
    name_b = st.selectbox("Actif B", keys, index=default_b, key="sel_b")
with ctrl3:
    capital = st.number_input("Capital ($)", value=1000, step=100)
with ctrl4:
    st.markdown("<div style='margin-top:22px'>", unsafe_allow_html=True)
    analyse = st.button("Analyser")
    st.markdown("</div>", unsafe_allow_html=True)

if name_a == name_b:
    st.warning("Choisis deux actifs différents.")
else:
    # Lancement de l'analyse au clic — stocke les résultats dans session_state
    if analyse:
        s_a, err_a = fetch_prices(CRYPTOS[name_a])
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
                beta = m["Hedge Ratio (β)"]
                p_a = float(s_a.iloc[-1])
                p_b = float(s_b.iloc[-1])
                ratio = abs(beta * p_b / p_a)
                alloc_a = capital / (1 + ratio)
                alloc_b = capital - alloc_a
                # Stocker tout ce qui est nécessaire au backtest
                st.session_state["bt_data"] = {
                    "m": m,
                    "name_a": name_a,
                    "name_b": name_b,
                    "alloc_a": alloc_a,
                    "alloc_b": alloc_b,
                }

    # Affichage du backtest — utilise les données en cache, se recalcule à chaque interaction
    if "bt_data" in st.session_state and st.session_state["bt_data"]["name_a"] == name_a and st.session_state["bt_data"]["name_b"] == name_b:
        bt = st.session_state["bt_data"]
        m        = bt["m"]
        alloc_a  = bt["alloc_a"]
        alloc_b  = bt["alloc_b"]

        st.markdown("#### Backtest")

        bp1, bp2, bp3 = st.columns(3)
        with bp1:
            entry_z = st.number_input("Seuil d'entrée (z)", value=2.0, step=0.1, min_value=0.5, max_value=5.0, key="bt_entry")
        with bp2:
            exit_z  = st.number_input("Seuil de sortie (z)", value=0.5, step=0.1, min_value=0.0, max_value=2.0, key="bt_exit")
        with bp3:
            stop_z  = st.number_input("Stop-loss (z)", value=3.5, step=0.1, min_value=2.0, max_value=6.0, key="bt_stop")

        z_score_series = m["z_score"].dropna()
        df_prices = m["df"]
        trades = []
        position = None

        for date, z_val in z_score_series.items():
            if date not in df_prices.index:
                continue
            price_a = df_prices.loc[date, "A"]
            price_b = df_prices.loc[date, "B"]

            if position is None:
                if z_val > entry_z:
                    units_a = alloc_a / price_a
                    units_b = alloc_b / price_b
                    position = {
                        "type": f"SHORT {name_a} / LONG {name_b}",
                        "entry_date": date, "entry_z": z_val,
                        "entry_price_a": price_a, "entry_price_b": price_b,
                        "units_a": units_a, "units_b": units_b,
                        "direction": "short_a",
                    }
                elif z_val < -entry_z:
                    units_a = alloc_a / price_a
                    units_b = alloc_b / price_b
                    position = {
                        "type": f"LONG {name_a} / SHORT {name_b}",
                        "entry_date": date, "entry_z": z_val,
                        "entry_price_a": price_a, "entry_price_b": price_b,
                        "units_a": units_a, "units_b": units_b,
                        "direction": "long_a",
                    }
            else:
                exit_normal = (
                    (position["direction"] == "short_a" and z_val < exit_z) or
                    (position["direction"] == "long_a"  and z_val > -exit_z)
                )
                exit_stop = abs(z_val) > stop_z
                if exit_normal or exit_stop:
                    if position["direction"] == "short_a":
                        pnl_a = (position["entry_price_a"] - price_a) * position["units_a"]
                        pnl_b = (price_b - position["entry_price_b"]) * position["units_b"]
                    else:
                        pnl_a = (price_a - position["entry_price_a"]) * position["units_a"]
                        pnl_b = (position["entry_price_b"] - price_b) * position["units_b"]
                    pnl_total = pnl_a + pnl_b
                    trades.append({
                        "entrée":    position["entry_date"].strftime("%Y-%m-%d"),
                        "sortie":    date.strftime("%Y-%m-%d"),
                        "type":      position["type"],
                        "z entrée":  round(position["entry_z"], 2),
                        "z sortie":  round(z_val, 2),
                        "P&L A ($)": round(pnl_a, 2),
                        "P&L B ($)": round(pnl_b, 2),
                        "P&L ($)":   round(pnl_total, 2),
                        "raison":    "Stop-loss" if exit_stop else "Retour à la moyenne",
                        "résultat":  "✅ Gagnant" if pnl_total > 0 else "❌ Perdant",
                    })
                    position = None

        if not trades:
            st.info("Aucun trade déclenché sur la période avec ces paramètres.")
        else:
            df_trades  = pd.DataFrame(trades)
            n_trades   = len(df_trades)
            n_win      = (df_trades["P&L ($)"] > 0).sum()
            win_rate   = n_win / n_trades
            pnl_values = df_trades["P&L ($)"].cumsum().tolist()
            total_pnl  = pnl_values[-1]
            cummax     = pd.Series(pnl_values).cummax()
            max_dd     = (pd.Series(pnl_values) - cummax).min()
            rets       = df_trades["P&L ($)"]
            sharpe     = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("P&L cumulé", f"{total_pnl:+.0f}$")
            b2.metric("Trades", n_trades)
            b3.metric("Win rate", f"{win_rate:.0%}")
            b4.metric("Drawdown max", f"{max_dd:.0f}$")
            b5.metric("Sharpe", f"{sharpe:.2f}")

            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=list(range(1, n_trades + 1)), y=pnl_values,
                mode="lines+markers",
                line=dict(color="#1D9E75", width=1.5),
                marker=dict(size=7, color=["#1D9E75" if p > 0 else "#E24B4A" for p in df_trades["P&L ($)"]])
            ))
            fig_pnl.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1)
            fig_pnl.update_layout(
                title=dict(text="P&L cumulé par trade (en $)", font=dict(size=12)),
                height=220, margin=dict(t=36, b=16, l=40, r=16),
                plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
            )
            fig_pnl.update_xaxes(title_text="Trade #", showgrid=False, tickfont=dict(size=10))
            fig_pnl.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
            st.plotly_chart(fig_pnl, use_container_width=True)

            with st.expander(f"Détail des {n_trades} trades"):
                st.dataframe(df_trades, use_container_width=True, hide_index=True)

        st.divider()
        df = m["df"]

        # Préparer les dates et valeurs des marqueurs si des trades existent
        if trades:
            entry_dates = pd.to_datetime(df_trades["entrée"])
            exit_dates  = pd.to_datetime(df_trades["sortie"])
            entry_colors = ["#1D9E75" if t.startswith("LONG") else "#E24B4A" for t in df_trades["type"]]
            exit_colors  = ["#E24B4A" if t.startswith("LONG") else "#1D9E75" for t in df_trades["type"]]

        # Graphe 1 — prix normalisés
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["A"]/df["A"].iloc[0], name=name_a, line=dict(color="#1D9E75", width=1.5)))
        fig.add_trace(go.Scatter(x=df.index, y=df["B"]/df["B"].iloc[0], name=name_b, line=dict(color="#7F77DD", width=1.5)))

        if trades:
            # Marqueurs d'entrée sur prix normalisé de A
            entry_y_a = [df["A"].loc[d] / df["A"].iloc[0] if d in df.index else None for d in entry_dates]
            exit_y_a  = [df["A"].loc[d] / df["A"].iloc[0] if d in df.index else None for d in exit_dates]
            fig.add_trace(go.Scatter(
                x=entry_dates, y=entry_y_a, mode="markers", name="Entrée",
                marker=dict(symbol="triangle-up", size=10, color=entry_colors, line=dict(width=1, color="#fff")),
                hovertemplate="Entrée %{x}<extra></extra>"
            ))
            fig.add_trace(go.Scatter(
                x=exit_dates, y=exit_y_a, mode="markers", name="Sortie",
                marker=dict(symbol="triangle-down", size=10, color=exit_colors, line=dict(width=1, color="#fff")),
                hovertemplate="Sortie %{x}<extra></extra>"
            ))

        fig.update_layout(
            title=dict(text="Prix normalisés (base 1)", font=dict(size=12)),
            height=260, margin=dict(t=36, b=16, l=40, r=16),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(size=11))
        )
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
        fig.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
        st.plotly_chart(fig, use_container_width=True)

        # Graphe 2 — z-score
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, line=dict(color="#378ADD", width=1.5), fill="tozeroy", fillcolor="rgba(55,138,221,0.05)", showlegend=False))
        for y_val, color in [(2, "rgba(220,50,50,0.5)"), (-2, "rgba(220,50,50,0.5)"), (0, "rgba(180,180,180,0.5)")]:
            fig2.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1)

        if trades:
            entry_z_vals = [z_score_series.loc[d] if d in z_score_series.index else None for d in entry_dates]
            exit_z_vals  = [z_score_series.loc[d] if d in z_score_series.index else None for d in exit_dates]
            fig2.add_trace(go.Scatter(
                x=entry_dates, y=entry_z_vals, mode="markers", name="Entrée",
                marker=dict(symbol="triangle-up", size=10, color=entry_colors, line=dict(width=1, color="#fff")),
                hovertemplate="Entrée %{x} · z=%{y:.2f}<extra></extra>"
            ))
            fig2.add_trace(go.Scatter(
                x=exit_dates, y=exit_z_vals, mode="markers", name="Sortie",
                marker=dict(symbol="triangle-down", size=10, color=exit_colors, line=dict(width=1, color="#fff")),
                hovertemplate="Sortie %{x} · z=%{y:.2f}<extra></extra>"
            ))

        fig2.update_layout(
            title=dict(text="Z-Score — signal de trading", font=dict(size=12)),
            height=260, margin=dict(t=36, b=16, l=40, r=16),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(size=11))
        )
        fig2.update_xaxes(showgrid=False, tickfont=dict(size=10))
        fig2.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
        st.plotly_chart(fig2, use_container_width=True)
