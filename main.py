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
/* Bordure pointillés + espacement uniforme sur les graphes Plotly */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
.stPlotlyChart {
    margin-bottom: 24px !important;
}
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

# ─── COULEURS PAR TOKEN ────────────────────────────────────────────────────────
TOKEN_COLORS = {
    "Bitcoin":      "#F7931A",
    "Ethereum":     "#8C8C8C",
    "Hyperliquid":  "rgb(151, 252, 228)",
}

def token_color(name: str) -> str:
    return TOKEN_COLORS.get(name, "#888888")

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

# ── Backtest ──────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Backtest")
st.caption("Capital par défaut : 1 000 $")

keys = list(CRYPTOS.keys())
default_a = keys.index(st.session_state.prefill_a) if st.session_state.prefill_a in keys else 0
default_b = keys.index(st.session_state.prefill_b) if st.session_state.prefill_b in keys else min(1, len(keys) - 1)

# Ligne 1 : paires + bouton
r1c1, r1c2, r1c3, _ = st.columns([1.0, 1.0, 0.4, 2.6])
with r1c1:
    name_a = st.selectbox("Actif A", keys, index=default_a, key="sel_a")
with r1c2:
    name_b = st.selectbox("Actif B", keys, index=default_b, key="sel_b")
with r1c3:
    st.markdown("<div style='margin-top:22px'>", unsafe_allow_html=True)
    analyse = st.button("Analyser")
    st.markdown("</div>", unsafe_allow_html=True)

capital = 1000

# Ligne 2 : paramètres backtest
r2c1, r2c2, r2c3, _ = st.columns([1.0, 1.0, 1.0, 2.0])
with r2c1:
    entry_z = st.number_input("Entrée (z)", value=2.0, step=0.1, min_value=0.5, max_value=5.0, key="bt_entry")
with r2c2:
    exit_z = st.number_input("Sortie (z)", value=0.5, step=0.1, min_value=0.0, max_value=2.0, key="bt_exit")
with r2c3:
    stop_z = st.number_input("Stop (z)", value=3.5, step=0.1, min_value=2.0, max_value=6.0, key="bt_stop")

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
                        "#":                  len(trades) + 1,
                        "entrée":             position["entry_date"].strftime("%Y-%m-%d"),
                        "sortie":             date.strftime("%Y-%m-%d"),
                        "durée (j)":          (date - position["entry_date"]).days,
                        "type":               position["type"],
                        "z entrée":           round(position["entry_z"], 2),
                        "z sortie":           round(z_val, 2),
                        f"{name_a} entrée $": round(position["entry_price_a"], 4),
                        f"{name_a} sortie $": round(price_a, 4),
                        f"{name_b} entrée $": round(position["entry_price_b"], 4),
                        f"{name_b} sortie $": round(price_b, 4),
                        f"P&L {name_a} ($)":  round(pnl_a, 2),
                        f"P&L {name_b} ($)":  round(pnl_b, 2),
                        "P&L ($)":            round(pnl_total, 2),
                        "raison":             "Stop-loss" if exit_stop else "Retour à la moyenne",
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

            pnl_color  = "#0F6E56" if total_pnl >= 0 else "#A32D2D"
            pnl_icon   = "▲" if total_pnl >= 0 else "▼"
            wr_color   = "#0F6E56" if win_rate >= 0.5 else "#A32D2D"
            wr_icon    = "▲" if win_rate >= 0.5 else "▼"
            dd_color   = "#A32D2D" if max_dd < 0 else "#0F6E56"
            sh_color   = "#0F6E56" if sharpe >= 1 else ("#854F0B" if sharpe >= 0 else "#A32D2D")
            sh_icon    = "▲" if sharpe >= 1 else ("—" if sharpe >= 0 else "▼")

            bt_cards = [
                ("P&L cumulé",   f"<span style='color:{pnl_color}'>{pnl_icon} {total_pnl:+.0f}$</span>"),
                ("Trades",       f"<span style='color:#333'>{n_trades}</span>"),
                ("Win rate",     f"<span style='color:{wr_color}'>{wr_icon} {win_rate:.0%}</span>"),
                ("Drawdown max", f"<span style='color:{dd_color}'>▼ {max_dd:.0f}$</span>"),
                ("Sharpe",       f"<span style='color:{sh_color}'>{sh_icon} {sharpe:.2f}</span>"),
            ]
            st.markdown("<div style='margin:16px 0 6px'></div>", unsafe_allow_html=True)
            bt_cols = st.columns(5)
            for col, (label, value) in zip(bt_cols, bt_cards):
                with col:
                    st.markdown(
                        f"""<div style="border:1px dashed #ccc;border-radius:8px;padding:12px 14px 10px;">
                        <p style="font-size:10px;color:#aaa;margin:0 0 6px">{label}</p>
                        <p style="font-size:20px;font-weight:500;margin:0">{value}</p>
                        </div>""",
                        unsafe_allow_html=True
                    )
            st.markdown("<div style='margin:16px 0 0'></div>", unsafe_allow_html=True)

            fig_pnl = go.Figure()

            # Ligne colorée selon positif/négatif — on découpe en segments
            x_vals = list(range(1, n_trades + 1))
            for i in range(len(pnl_values) - 1):
                y0, y1 = pnl_values[i], pnl_values[i + 1]
                # Si le segment croise zéro, on interpole le point de croisement
                if (y0 >= 0 and y1 >= 0):
                    color = "#1D9E75"
                    fig_pnl.add_trace(go.Scatter(x=[x_vals[i], x_vals[i+1]], y=[y0, y1],
                        mode="lines", line=dict(color=color, width=2), showlegend=False))
                elif (y0 < 0 and y1 < 0):
                    color = "#E24B4A"
                    fig_pnl.add_trace(go.Scatter(x=[x_vals[i], x_vals[i+1]], y=[y0, y1],
                        mode="lines", line=dict(color=color, width=2), showlegend=False))
                else:
                    # Croisement de zéro — interpolation
                    t = y0 / (y0 - y1)
                    x_cross = x_vals[i] + t * (x_vals[i+1] - x_vals[i])
                    c1 = "#1D9E75" if y0 >= 0 else "#E24B4A"
                    c2 = "#E24B4A" if y0 >= 0 else "#1D9E75"
                    fig_pnl.add_trace(go.Scatter(x=[x_vals[i], x_cross], y=[y0, 0],
                        mode="lines", line=dict(color=c1, width=2), showlegend=False))
                    fig_pnl.add_trace(go.Scatter(x=[x_cross, x_vals[i+1]], y=[0, y1],
                        mode="lines", line=dict(color=c2, width=2), showlegend=False))

            # Marqueurs avec valeur P&L affichée
            marker_colors = ["#1D9E75" if p > 0 else "#E24B4A" for p in pnl_values]
            marker_labels = [f"+{p:.0f}$" if p > 0 else f"{p:.0f}$" for p in pnl_values]
            fig_pnl.add_trace(go.Scatter(
                x=x_vals, y=pnl_values, mode="markers+text",
                text=marker_labels,
                textposition=["top center" if p >= 0 else "bottom center" for p in pnl_values],
                textfont=dict(size=9, color=marker_colors),
                marker=dict(size=7, color=marker_colors),
                showlegend=False,
                hovertemplate="Trade #%{x}<br>P&L cumulé : %{y:.0f}$<extra></extra>"
            ))

            fig_pnl.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1)
            pnl_abs_max = max(abs(min(pnl_values)), abs(max(pnl_values))) * 1.25
            # Tableau détail trades — au dessus des graphes
            with st.expander(f"Détail des {n_trades} trades"):
                st.markdown(
                    f"<p style='font-size:12px;color:#666;margin:0 0 10px'>"
                    f"Beta (Hedge Ratio) : {m['Hedge Ratio (β)']:.4f} — "
                    f"pour {capital}$, allouer <strong>{alloc_a:.0f}$</strong> sur {name_a} "
                    f"et <strong>{alloc_b:.0f}$</strong> sur {name_b}.</p>",
                    unsafe_allow_html=True
                )
                df_display = df_trades.copy()
                df_display["type"] = df_display["type"].apply(lambda t:
                    t.replace(f"LONG {name_a}", f"↑ {name_a}")
                     .replace(f"SHORT {name_a}", f"↓ {name_a}")
                     .replace(f"LONG {name_b}", f"↑ {name_b}")
                     .replace(f"SHORT {name_b}", f"↓ {name_b}")
                )
                def _row_color(row):
                    try:
                        v = float(row["P&L ($)"])
                        if v > 0: return ["background-color:#e8f7f1;color:#0F6E56"] * len(row)
                        if v < 0: return ["background-color:#fdf0f0;color:#A32D2D"] * len(row)
                    except: pass
                    return [""] * len(row)

                pnl_a_col = f"P&L {name_a} ($)"
                pnl_b_col = f"P&L {name_b} ($)"

                def _color_pnl_leg(val):
                    try:
                        v = float(val)
                        if v > 0: return "color:#0F6E56;font-weight:700"
                        if v < 0: return "color:#A32D2D;font-weight:700"
                    except: pass
                    return "font-weight:700"

                def _fmt_pnl(v):
                    try:
                        return f"{float(v):+.2f}$"
                    except:
                        return v

                fmt_dict = {"P&L ($)": _fmt_pnl}
                if pnl_a_col in df_display.columns:
                    fmt_dict[pnl_a_col] = _fmt_pnl
                    fmt_dict[pnl_b_col] = _fmt_pnl

                styled = df_display.style.apply(_row_color, axis=1).format(fmt_dict)
                if pnl_a_col in df_display.columns:
                    styled = styled.applymap(_color_pnl_leg, subset=[pnl_a_col, pnl_b_col, "P&L ($)"])

                st.dataframe(styled, use_container_width=True, hide_index=True)

            fig_pnl.update_layout(
                title=dict(text="P&L cumulé par trade (en $)", font=dict(size=12)),
                height=260, margin=dict(t=40, b=28, l=48, r=24),
                plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
                yaxis=dict(range=[-pnl_abs_max, pnl_abs_max]),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                             line=dict(color="#ccc", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")]
            )
            fig_pnl.update_xaxes(title_text="", showgrid=False, tickfont=dict(size=10))
            fig_pnl.update_yaxes(showgrid=False, tickfont=dict(size=10))
            st.plotly_chart(fig_pnl, use_container_width=True)

        df = m["df"]

        # Plage de dates commune pour aligner les axes des 2 graphes
        x_min = df.index.min()
        x_max = df.index.max()

        # Préparer les dates et valeurs des marqueurs si des trades existent
        if trades:
            entry_dates  = pd.to_datetime(df_trades["entrée"])
            exit_dates   = pd.to_datetime(df_trades["sortie"])
            entry_colors = ["#1D9E75" if t.startswith("LONG") else "#E24B4A" for t in df_trades["type"]]
            exit_colors  = ["#E24B4A" if t.startswith("LONG") else "#1D9E75" for t in df_trades["type"]]
            trade_nums   = list(range(1, n_trades + 1))

            # Prix réels aux dates d'entrée et sortie
            entry_pa = [df_prices.loc[d, "A"] if d in df_prices.index else None for d in entry_dates]
            entry_pb = [df_prices.loc[d, "B"] if d in df_prices.index else None for d in entry_dates]
            exit_pa  = [df_prices.loc[d, "A"] if d in df_prices.index else None for d in exit_dates]
            exit_pb  = [df_prices.loc[d, "B"] if d in df_prices.index else None for d in exit_dates]

            entry_hover = [
                f"<b>Trade #{n} — Entrée</b><br>{t}<br>Date : {d.strftime('%Y-%m-%d')}<br>"
                f"{name_a} : {pa:.4f}$<br>{name_b} : {pb:.4f}$<br>z : {ze:.2f}"
                for n, t, d, pa, pb, ze in zip(
                    trade_nums, df_trades["type"], entry_dates,
                    entry_pa, entry_pb, df_trades["z entrée"]
                )
            ]
            exit_hover = [
                f"<b>Trade #{n} — Sortie</b><br>{r}<br>Date : {d.strftime('%Y-%m-%d')}<br>"
                f"{name_a} : {pa:.4f}$<br>{name_b} : {pb:.4f}$<br>z : {zs:.2f}<br>"
                f"P&L : <b>{pnl:+.2f}$</b>"
                for n, r, d, pa, pb, zs, pnl in zip(
                    trade_nums, df_trades["raison"], exit_dates,
                    exit_pa, exit_pb, df_trades["z sortie"], df_trades["P&L ($)"]
                )
            ]

        # Graphe 1 — double axe Y avec prix réels
        color_a = token_color(name_a)
        color_b = token_color(name_b)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df.index, y=df["A"], name=name_a,
            line=dict(color=color_a, width=1.5),
            yaxis="y1"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, y=df["B"], name=name_b,
            line=dict(color=color_b, width=1.5),
            yaxis="y2"
        ))

        if trades:
            entry_y_a = [df["A"].loc[d] if d in df.index else None for d in entry_dates]
            exit_y_a  = [df["A"].loc[d] if d in df.index else None for d in exit_dates]

            fig.add_trace(go.Scatter(
                x=entry_dates, y=entry_y_a, mode="markers+text", name="Entrée",
                yaxis="y1",
                text=trade_nums, textposition="top center",
                textfont=dict(size=9, color="#555"),
                marker=dict(symbol="triangle-up", size=11, color=entry_colors, line=dict(width=1, color="#fff")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=entry_hover,
            ))
            fig.add_trace(go.Scatter(
                x=exit_dates, y=exit_y_a, mode="markers+text", name="Sortie",
                yaxis="y1",
                text=trade_nums, textposition="bottom center",
                textfont=dict(size=9, color="#555"),
                marker=dict(symbol="triangle-down", size=11, color=exit_colors, line=dict(width=1, color="#fff")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=exit_hover,
            ))

        fig.update_layout(
            title=dict(text=f"Évolution des prix — {name_a} (gauche) · {name_b} (droite)", font=dict(size=12)),
            height=260, margin=dict(t=40, b=28, l=56, r=56),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            showlegend=True,
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(size=11)),
            xaxis=dict(range=[x_min, x_max], showgrid=False, tickfont=dict(size=10)),
            yaxis=dict(
                title=dict(text=name_a, font=dict(color=color_a, size=10)),
                tickfont=dict(color=color_a, size=9),
                showgrid=False,
            ),
            yaxis2=dict(
                title=dict(text=name_b, font=dict(color=color_b, size=10)),
                tickfont=dict(color=color_b, size=9),
                overlaying="y", side="right",
                showgrid=False,
            ),
            shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                         line=dict(color="#ccc", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")]
        )
        st.plotly_chart(fig, use_container_width=True)

        # Graphe 2 — z-score
        fig2 = go.Figure()

        # Courbe z-score
        fig2.add_trace(go.Scatter(
            x=z_score_series.index, y=z_score_series,
            line=dict(color="#378ADD", width=1.5),
            fill="tozeroy", fillcolor="rgba(55,138,221,0.05)",
            showlegend=False
        ))

        # Lignes horizontales seuils
        fig2.add_hline(y=entry_z,  line_color="rgba(220,50,50,0.7)",  line_width=1.5)
        fig2.add_hline(y=-entry_z, line_color="rgba(220,50,50,0.7)",  line_width=1.5)
        fig2.add_hline(y=0,        line_color="rgba(180,180,180,0.6)", line_width=1, line_dash="dot")

        if trades:
            entry_z_vals = [z_score_series.loc[d] if d in z_score_series.index else None for d in entry_dates]
            exit_z_vals  = [z_score_series.loc[d] if d in z_score_series.index else None for d in exit_dates]

            fig2.add_trace(go.Scatter(
                x=entry_dates, y=entry_z_vals, mode="markers+text", name="Entrée",
                text=trade_nums, textposition="top center",
                textfont=dict(size=9, color="#555"),
                marker=dict(symbol="triangle-up", size=11, color=entry_colors, line=dict(width=1, color="#fff")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=entry_hover,
            ))
            fig2.add_trace(go.Scatter(
                x=exit_dates, y=exit_z_vals, mode="markers+text", name="Sortie",
                text=trade_nums, textposition="bottom center",
                textfont=dict(size=9, color="#555"),
                marker=dict(symbol="triangle-down", size=11, color=exit_colors, line=dict(width=1, color="#fff")),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=exit_hover,
            ))

        z_abs_max = max(abs(z_score_series.max()), abs(z_score_series.min())) * 1.15
        fig2.update_layout(
            title=dict(text="Z-Score — signal de trading", font=dict(size=12)),
            height=260, margin=dict(t=40, b=28, l=48, r=24),
            plot_bgcolor="#fff", paper_bgcolor="#fff",
            showlegend=False,
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[-z_abs_max, z_abs_max]),
            shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                         line=dict(color="#ccc", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")]
        )
        fig2.update_xaxes(showgrid=False, tickfont=dict(size=10))
        fig2.update_yaxes(showgrid=False, tickfont=dict(size=10))
        st.plotly_chart(fig2, use_container_width=True)
