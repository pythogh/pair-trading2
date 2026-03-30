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
[data-testid="stSelectbox"] > div > div { min-height: 32px !important; font-size: 12px !important; }
[data-testid="stNumberInput"] input { height: 32px !important; font-size: 12px !important; padding: 4px 8px !important; }
[data-testid="stDateInput"] input { height: 32px !important; font-size: 12px !important; padding: 4px 8px !important; }
[data-testid="stDateInput"] label { font-size: 11px !important; color: #888 !important; }
[data-testid="stAlert"] { font-size: 12px !important; padding: 8px 14px !important; }
button[data-baseweb="tab"] { font-size: 12px !important; padding: 8px 20px !important; margin-right: 4px !important; }
.stApp { background-color: #ffffff !important; }
/* Logos ronds dans les tableaux */
[data-testid="stDataFrame"] img { border-radius: 50% !important; }
/* Bordure pointillés + espacement uniforme sur les graphes Plotly */
[data-testid="stPlotlyChart"],
[data-testid="stPlotlyChart"] > div,
.stPlotlyChart {
    margin-bottom: 24px !important;
}
</style>
""", unsafe_allow_html=True)

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR    = "data-hourly"
BARS_PER_DAY = 24
Z_WINDOW     = 336   # ~14 jours en bougies horaires
HL_MAX_BARS  = 360   # ~15 jours en bougies horaires
MIN_BARS     = 240   # minimum 10 jours de données

def scan_tokens(data_dir):
    files = glob.glob(os.path.join(data_dir, "*-historical-data.csv"))
    tokens = {}
    for f in sorted(files):
        slug = os.path.basename(f).replace("-historical-data.csv", "")
        label = slug.replace("-", " ").title()
        tokens[label] = slug
    return tokens

CRYPTOS = scan_tokens(DATA_DIR)

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
def fetch_prices(slug, data_dir=DATA_DIR):
    path = os.path.join(data_dir, f"{slug}-historical-data.csv")
    if not os.path.exists(path):
        return None, f"Fichier introuvable : {path}"
    try:
        df = pd.read_csv(path, header=0)
        df.columns = [c.strip() for c in df.columns]
        date_col  = df.columns[0]
        price_col = df.columns[1]
        df[date_col]  = pd.to_datetime(df[date_col], dayfirst=True)
        df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
        df = df.sort_values(date_col).set_index(date_col)
        series = df[price_col].dropna()
        if len(series) < MIN_BARS:
            return None, f"{slug} : seulement {len(series)} bougies (minimum {MIN_BARS})"
        return series, None
    except Exception as e:
        return None, f"Erreur lecture {slug} : {str(e)}"


def compute_metrics(series_a, series_b, name_a, name_b):
    df = pd.concat([series_a, series_b], axis=1).dropna()
    df.columns = ["A", "B"]
    if len(df) < MIN_BARS:
        return None

    returns = df.pct_change().dropna()
    correlation = returns["A"].corr(returns["B"])

    X = sm.add_constant(df["B"])
    model = sm.OLS(df["A"], X).fit()
    beta  = model.params["B"]
    alpha = model.params["const"]

    spread = df["A"] - (beta * df["B"] + alpha)
    try:
        p_value = adfuller(spread.dropna())[1]
    except Exception:
        p_value = 1.0

    z_score   = (spread - spread.rolling(Z_WINDOW).mean()) / spread.rolling(Z_WINDOW).std()
    current_z = float(z_score.iloc[-1])

    spread_lag  = spread.shift(1)
    spread_diff = spread.diff()
    valid = ~(spread_lag.isna() | spread_diff.isna())
    try:
        res = sm.OLS(spread_diff[valid], spread_lag[valid]).fit()
        lambda_val     = -res.params.iloc[0]
        hl_bars        = np.log(2) / lambda_val if lambda_val > 0 else float("inf")
        hl_days        = hl_bars / BARS_PER_DAY
    except Exception:
        hl_bars = float("inf")
        hl_days = float("inf")

    ok_corr = correlation >= 0.7
    ok_p    = p_value < 0.05
    ok_hl   = hl_bars < HL_MAX_BARS
    ok_z    = abs(current_z) > 2

    verdict       = "✅ Valide" if (ok_corr and ok_p and ok_hl and ok_z) else "❌ Non valide"
    verdict_color = "green" if verdict == "✅ Valide" else "red"

    if current_z > 2:
        signal = f"SHORT ↓ {name_a} / LONG ↑ {name_b}"
    elif current_z < -2:
        signal = f"LONG ↑ {name_a} / SHORT ↓ {name_b}"
    else:
        signal = "—"

    hl_display = round(hl_days, 1) if hl_days != float("inf") else "∞"

    return {
        "Corrélation":        round(correlation, 3),
        "Hedge Ratio (β)":    round(float(beta), 4),
        "Co-intégration (p)": round(p_value, 4),
        "Half-Life (j)":      hl_display,
        "Z-Score":            round(current_z, 2),
        "Verdict":            verdict,
        "Signal":             signal,
        "spread":             spread,
        "z_score":            z_score,
        "df":                 df,
        "verdict_color":      verdict_color,
    }

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("📈 Pair Trading Analyzer")
st.caption(f"Données horaires · {len(CRYPTOS)} tokens · dossier `{DATA_DIR}/`")

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "prefill_a" not in st.session_state:
    st.session_state.prefill_a = None
if "prefill_b" not in st.session_state:
    st.session_state.prefill_b = None
if "matrix_results" not in st.session_state:
    st.session_state["matrix_results"] = []
if "token_logos" not in st.session_state:
    st.session_state["token_logos"] = {}

# ─── LOGOS + METADATA ──────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def fetch_token_metadata(slugs_map: tuple, api_key: str) -> dict:
    """
    Récupère logo, symbol et name depuis CMC.
    slugs_map : tuple de (label, cmc_slug)
    Retourne dict {label: {logo, symbol, name}}
    """
    import requests
    result = {label: {"logo": "", "symbol": "", "name": label} for label, _ in slugs_map}
    if not api_key:
        result["__error__"] = "Clé API_CMC manquante."
        return result
    try:
        cmc_slugs = [cmc_slug for _, cmc_slug in slugs_map]
        slug_to_label = {cmc_slug: label for label, cmc_slug in slugs_map}

        r = requests.get(
            "https://pro-api.coinmarketcap.com/v1/cryptocurrency/info",
            params={"slug": ",".join(cmc_slugs), "aux": "logo"},
            headers={"X-CMC_PRO_API_KEY": api_key, "Accept": "application/json"},
            timeout=15,
        )
        if r.status_code == 200:
            for coin in r.json().get("data", {}).values():
                cmc_slug = coin.get("slug", "")
                label = slug_to_label.get(cmc_slug, "")
                if label:
                    result[label] = {
                        "logo":   coin.get("logo", ""),
                        "symbol": coin.get("symbol", ""),
                        "name":   coin.get("name", label),
                    }
        else:
            result["__error__"] = f"HTTP {r.status_code}: {r.text[:300]}"
    except Exception as e:
        result["__error__"] = str(e)
    return result

def load_tokens_map() -> dict:
    """Utilise le slug du fichier CSV directement comme slug CMC."""
    return {label: slug for label, slug in CRYPTOS.items()}

TOKEN_CMC_MAP = load_tokens_map()

if not st.session_state["token_logos"]:
    _cmc_key = st.secrets["API_CMC"] if "API_CMC" in st.secrets else ""
    _slugs_map = tuple(TOKEN_CMC_MAP.items())
    with st.spinner("Chargement des métadonnées tokens…"):
        st.session_state["token_logos"] = fetch_token_metadata(_slugs_map, _cmc_key)

def get_logo(name: str) -> str:
    return st.session_state["token_logos"].get(name, {}).get("logo", "")

def get_display_name(name: str) -> str:
    """Retourne 'SYM · Name propre' ou le label par défaut."""
    meta = st.session_state["token_logos"].get(name, {})
    sym  = meta.get("symbol", "")
    cmc_name = meta.get("name", name)
    if sym and cmc_name:
        return f"{sym} · {cmc_name}"
    return name

def dn(name: str) -> str:
    """Display name propre depuis CMC, ou label par défaut."""
    meta = st.session_state.get("token_logos", {}).get(name, {})
    cmc_name = meta.get("name", "")
    return cmc_name if cmc_name else name

def logo_html(name: str, size: int = 18) -> str:
    url = get_logo(name)
    if not url:
        return f"<span style='display:inline-block;width:{size}px;height:{size}px;border-radius:50%;background:#e0e0e0;'></span>"
    return (f"<img src='{url}' style='width:{size}px;height:{size}px;"
            f"border-radius:50%;object-fit:cover;vertical-align:middle;'>")

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
            "Paire":           f"{dn(a)} / {dn(b)}",
            "Corrélation":     m["Corrélation"],
            "Hedge Ratio β":   m["Hedge Ratio (β)"],
            "Co-intégration p": m["Co-intégration (p)"],
            "Half-Life":       m["Half-Life (j)"],
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
        "seuil": "5–15 jours (120–360h)",
        "formule": "t<sub>½</sub> = ln(2) / λ &nbsp;&nbsp; Δs<sub>t</sub> = λ · s<sub>t−1</sub>",
        "note": "Calculé sur bougies horaires, affiché en jours. Fenêtre z-score = 336h (14j).",
    },
    {
        "emoji": "🌡️",
        "name": "Z-Score",
        "seuil": "Signal si |z| > 2",
        "formule": "z = (s<sub>t</sub> − μ<sub>30</sub>) / σ<sub>30</sub>",
        "note": "Fenêtre glissante 14 jours (336h). Un z > +2 se produit ~2.5% du temps — signal de trading.",
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
            .format({
                "Corrélation":      "{:.3f}",
                "Hedge Ratio β":    "{:.4f}",
                "Co-intégration p": "{:.4f}",
                "Z-Score":          "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# ── Paramètres globaux ────────────────────────────────────────────────────────
import datetime as dt
st.markdown("<p style='font-size:11px;color:#aaa;margin:8px 0 4px'>Paramètres de stratégie (données horaires)</p>", unsafe_allow_html=True)
gp1, gp2, gp3, gp4, gp5 = st.columns([0.4, 0.4, 0.4, 0.4, 2.4])
with gp1:
    entry_z = st.number_input("Entrée z", value=2.0, step=0.1, min_value=0.5, max_value=5.0, key="bt_entry")
with gp2:
    exit_z = st.number_input("Sortie z", value=0.5, step=0.1, min_value=0.0, max_value=2.0, key="bt_exit")
with gp3:
    stop_z = st.number_input("Stop z", value=3.5, step=0.1, min_value=2.0, max_value=6.0, key="bt_stop")
with gp4:
    max_duration = st.number_input("Durée max (h)", value=72, step=6, min_value=6, max_value=720, key="bt_duration",
                                   help="Durée maximale d'un trade en heures (ex: 72h = 3 jours)")
with gp5:
    today = dt.date.today()
    period_days = st.slider(
        "Période d'analyse",
        min_value=7, max_value=90, value=90, step=7,
        format="%d j",
        key="bt_period"
    )
    date_start = today - dt.timedelta(days=period_days)
    date_end   = today
    st.caption(f"{date_start.strftime('%d %b %Y')} → {date_end.strftime('%d %b %Y')}")

ts_start = pd.Timestamp(date_start)
ts_end   = pd.Timestamp(date_end)

# ── Onglets Backtest / Winrate ────────────────────────────────────────────────
tab_wr, tab_bt, tab_logo = st.tabs(["🏆 Win Rate", "🔍 Backtest", "🧪 Test Logo"])

with tab_bt:
    st.caption("Capital par défaut : 1 000 $")

    keys = list(CRYPTOS.keys())
    default_a = keys.index(st.session_state.prefill_a) if st.session_state.prefill_a in keys else 0
    default_b = keys.index(st.session_state.prefill_b) if st.session_state.prefill_b in keys else min(1, len(keys) - 1)

    c1, c2, c3, _ = st.columns([0.8, 0.8, 0.35, 2.05])
    with c1:
        name_a = st.selectbox("Actif A", keys, index=default_a, key="sel_a")
    with c2:
        name_b = st.selectbox("Actif B", keys, index=default_b, key="sel_b")
    with c3:
        st.markdown("<div style='margin-top:22px'>", unsafe_allow_html=True)
        analyse = st.button("Analyser")
        st.markdown("</div>", unsafe_allow_html=True)

    capital = 1000

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

            # Filtrer sur la période sélectionnée
            z_score_series = z_score_series[(z_score_series.index >= ts_start) & (z_score_series.index <= ts_end)]
            df_prices = df_prices[(df_prices.index >= ts_start) & (df_prices.index <= ts_end)]

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
                            "type": f"SHORT {dn(name_a)} / LONG {dn(name_b)}",
                            "entry_date": date, "entry_z": z_val,
                            "entry_price_a": price_a, "entry_price_b": price_b,
                            "units_a": units_a, "units_b": units_b,
                            "direction": "short_a",
                        }
                    elif z_val < -entry_z:
                        units_a = alloc_a / price_a
                        units_b = alloc_b / price_b
                        position = {
                            "type": f"LONG {dn(name_a)} / SHORT {dn(name_b)}",
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
                    exit_stop     = abs(z_val) > stop_z
                    exit_duration = (date - position["entry_date"]).days >= max_duration
                    if exit_normal or exit_stop or exit_duration:
                        raison = "Stop-loss" if exit_stop else ("Durée max" if exit_duration else "Retour à la moyenne")
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
                            f"{dn(name_a)} entrée $": round(position["entry_price_a"], 4),
                            f"{dn(name_a)} sortie $": round(price_a, 4),
                            f"{dn(name_b)} entrée $": round(position["entry_price_b"], 4),
                            f"{dn(name_b)} sortie $": round(price_b, 4),
                            f"P&L {dn(name_a)} ($)":  round(pnl_a, 2),
                            f"P&L {dn(name_b)} ($)":  round(pnl_b, 2),
                            "P&L ($)":            round(pnl_total, 2),
                            "raison":             raison,
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

                # Tableau détail trades
                with st.expander(f"Détail des {n_trades} trades", expanded=True):
                    st.markdown(
                        f"<p style='font-size:12px;color:#666;margin:0 0 10px'>"
                        f"Beta (Hedge Ratio) : {m['Hedge Ratio (β)']:.4f} — "
                        f"pour {capital}$, allouer <strong>{alloc_a:.0f}$</strong> sur {name_a} "
                        f"et <strong>{alloc_b:.0f}$</strong> sur {name_b}.</p>",
                        unsafe_allow_html=True
                    )
                    df_display = df_trades.copy()
                    df_display["type"] = df_display["type"].apply(lambda t:
                        t.replace(f"LONG {dn(name_a)}", f"↑ {dn(name_a)}")
                         .replace(f"SHORT {dn(name_a)}", f"↓ {dn(name_a)}")
                         .replace(f"LONG {dn(name_b)}", f"↑ {dn(name_b)}")
                         .replace(f"SHORT {dn(name_b)}", f"↓ {dn(name_b)}")
                    )
                    def _row_color(row):
                        try:
                            v = float(row["P&L ($)"])
                            if v > 0: return ["background-color:#e8f7f1;color:#0F6E56"] * len(row)
                            if v < 0: return ["background-color:#fdf0f0;color:#A32D2D"] * len(row)
                        except: pass
                        return [""] * len(row)

                    pnl_a_col = f"P&L {dn(name_a)} ($)"
                    pnl_b_col = f"P&L {dn(name_b)} ($)"

                    def _color_pnl_leg(val):
                        try:
                            v = float(val)
                            if v > 0: return "color:#0F6E56;font-weight:700"
                            if v < 0: return "color:#A32D2D;font-weight:700"
                        except: pass
                        return "font-weight:700"

                    def _fmt_pnl(v):
                        try: return f"{float(v):+.2f}$"
                        except: return v

                    fmt_dict = {"P&L ($)": _fmt_pnl}
                    if pnl_a_col in df_display.columns:
                        fmt_dict[pnl_a_col] = _fmt_pnl
                        fmt_dict[pnl_b_col] = _fmt_pnl

                    styled = df_display.style.apply(_row_color, axis=1).format(fmt_dict)
                    if pnl_a_col in df_display.columns:
                        styled = styled.applymap(_color_pnl_leg, subset=[pnl_a_col, pnl_b_col, "P&L ($)"])
                    st.dataframe(styled, use_container_width=True, hide_index=True)

            df = m["df"]
            df = df[(df.index >= ts_start) & (df.index <= ts_end)]

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
                    f"{dn(name_a)} : {pa:.4f}$<br>{dn(name_b)} : {pb:.4f}$<br>z : {ze:.2f}"
                    for n, t, d, pa, pb, ze in zip(
                        trade_nums, df_trades["type"], entry_dates,
                        entry_pa, entry_pb, df_trades["z entrée"]
                    )
                ]
                exit_hover = [
                    f"<b>Trade #{n} — Sortie</b><br>{r}<br>Date : {d.strftime('%Y-%m-%d')}<br>"
                    f"{dn(name_a)} : {pa:.4f}$<br>{dn(name_b)} : {pb:.4f}$<br>z : {zs:.2f}<br>"
                    f"P&L : <b>{pnl:+.2f}$</b>"
                    for n, r, d, pa, pb, zs, pnl in zip(
                        trade_nums, df_trades["raison"], exit_dates,
                        exit_pa, exit_pb, df_trades["z sortie"], df_trades["P&L ($)"]
                    )
                ]

            # Graphe 1 — double axe Y avec prix réels
            color_a = "#1B4F8A"   # bleu foncé
            color_b = "#5BA4CF"   # bleu clair
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["A"], name=dn(name_a),
                line=dict(color=color_a, width=1.5),
                yaxis="y1"
            ))
            fig.add_trace(go.Scatter(
                x=df.index, y=df["B"], name=dn(name_b),
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
                title=dict(text=f"Évolution des prix — {dn(name_a)} · {dn(name_b)}", font=dict(size=12)),
                height=260, margin=dict(t=40, b=28, l=48, r=48),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                showlegend=False,
                xaxis=dict(range=[x_min, x_max], showgrid=False, tickfont=dict(size=10)),
                yaxis=dict(showgrid=False, tickfont=dict(color=color_a, size=9), title=None),
                yaxis2=dict(
                    overlaying="y", side="right",
                    showgrid=False,
                    tickfont=dict(color=color_b, size=9),
                    title=None,
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

            # Lignes horizontales seuils — entry_z paramétrable + z=2/-2 fixes
            fig2.add_hline(y=entry_z,  line_color="rgba(220,50,50,0.7)",  line_width=1.5)
            fig2.add_hline(y=-entry_z, line_color="rgba(220,50,50,0.7)",  line_width=1.5)
            if entry_z != 2.0:
                fig2.add_hline(y=2,  line_color="rgba(220,50,50,0.3)", line_width=1, line_dash="dot")
                fig2.add_hline(y=-2, line_color="rgba(220,50,50,0.3)", line_width=1, line_dash="dot")
            fig2.add_hline(y=0, line_color="rgba(180,180,180,0.6)", line_width=1, line_dash="dot")

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
                height=260, margin=dict(t=40, b=28, l=48, r=48),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                showlegend=False,
                xaxis=dict(range=[x_min, x_max]),
                yaxis=dict(range=[-z_abs_max, z_abs_max], mirror=True, tickfont=dict(size=10)),
                shapes=[dict(type="rect", xref="paper", yref="paper", x0=0, y0=0, x1=1, y1=1,
                             line=dict(color="#ccc", width=1, dash="dot"), fillcolor="rgba(0,0,0,0)")]
            )
            fig2.update_xaxes(showgrid=False, tickfont=dict(size=10))
            fig2.update_yaxes(showgrid=False)
            st.plotly_chart(fig2, use_container_width=True)

with tab_wr:
    st.caption("Win rate de chaque paire calculé avec les paramètres backtest actuels.")

    if st.button("Calculer la matrice", use_container_width=False):
        all_names = list(CRYPTOS.keys())
        n = len(all_names)

        _progress_container = st.empty()
        bar = _progress_container.progress(0, text="Calcul en cours...")
        price_cache = {}
        for name in all_names:
            price_cache[name], _ = fetch_prices(CRYPTOS[name])

        wr_matrix = pd.DataFrame(index=all_names, columns=all_names, dtype=object)
        total_pairs = n * (n - 1) // 2
        done = 0
        for i, a in enumerate(all_names):
            for j, b in enumerate(all_names):
                if i >= j:
                    wr_matrix.loc[a, b] = None
                    continue
                done += 1
                bar.progress(done / total_pairs, text=f"{dn(a)} / {dn(b)}…")
                sa = price_cache.get(a)
                sb = price_cache.get(b)
                if sa is None or sb is None:
                    wr_matrix.loc[a, b] = None
                    wr_matrix.loc[b, a] = None
                    continue
                m_pair = compute_metrics(sa, sb, a, b)
                if m_pair is None:
                    wr_matrix.loc[a, b] = None
                    wr_matrix.loc[b, a] = None
                    continue

                z_s = m_pair["z_score"].dropna()
                df_p = m_pair["df"]
                beta_p = m_pair["Hedge Ratio (β)"]
                pa_last = float(sa.iloc[-1])
                pb_last = float(sb.iloc[-1])
                ratio_p = abs(beta_p * pb_last / pa_last)
                alloc_a_p = 1000 / (1 + ratio_p)
                alloc_b_p = 1000 - alloc_a_p

                trades_p, pos_p = [], None
                for date, z_val in z_s.items():
                    if date not in df_p.index:
                        continue
                    pa = df_p.loc[date, "A"]
                    pb = df_p.loc[date, "B"]
                    if pos_p is None:
                        if z_val > entry_z:
                            pos_p = {"dir": "short_a", "ed": date, "epa": pa, "epb": pb,
                                     "ua": alloc_a_p/pa, "ub": alloc_b_p/pb}
                        elif z_val < -entry_z:
                            pos_p = {"dir": "long_a", "ed": date, "epa": pa, "epb": pb,
                                     "ua": alloc_a_p/pa, "ub": alloc_b_p/pb}
                    else:
                        ex_n = (pos_p["dir"] == "short_a" and z_val < exit_z) or \
                               (pos_p["dir"] == "long_a"  and z_val > -exit_z)
                        ex_s = abs(z_val) > stop_z
                        ex_d = (date - pos_p["ed"]).days >= max_duration
                        if ex_n or ex_s or ex_d:
                            if pos_p["dir"] == "short_a":
                                pnl = (pos_p["epa"] - pa) * pos_p["ua"] + (pb - pos_p["epb"]) * pos_p["ub"]
                            else:
                                pnl = (pa - pos_p["epa"]) * pos_p["ua"] + (pos_p["epb"] - pb) * pos_p["ub"]
                            trades_p.append(pnl)
                            pos_p = None

                if not trades_p:
                    wr_matrix.loc[a, b] = None
                    wr_matrix.loc[b, a] = None
                else:
                    wr = sum(1 for p in trades_p if p > 0) / len(trades_p)
                    wr_matrix.loc[a, b] = wr
                    wr_matrix.loc[b, a] = wr

        _progress_container.empty()
        # Stocker en session_state pour persistance + signature des paramètres
        st.session_state["wr_matrix"] = wr_matrix.to_dict()
        st.session_state["wr_labels"] = all_names
        st.session_state["wr_params"] = (entry_z, exit_z, stop_z, max_duration, str(ts_start), str(ts_end))

    # Avertissement si paramètres changés depuis le dernier calcul
    if "wr_matrix" in st.session_state:
        current_params = (entry_z, exit_z, stop_z, max_duration, str(ts_start), str(ts_end))
        if st.session_state.get("wr_params") != current_params:
            st.warning("⚠️ Les paramètres ont changé — recalcule la matrice pour mettre à jour.")
    # Affichage — depuis session_state si disponible
    if "wr_matrix" in st.session_state:
        labels = st.session_state["wr_labels"]
        wr_matrix = pd.DataFrame(st.session_state["wr_matrix"])

        # Filtre sur le Win Rate — réduit la matrice
        wr_min = st.slider("Afficher uniquement les paires avec Win Rate ≥", 0, 100, 80, 5, format="%d%%")

        # Garder seulement les tokens qui ont au moins une paire au-dessus du seuil
        threshold = wr_min / 100
        tokens_to_keep = set()
        for a in labels:
            for b in labels:
                if a == b:
                    continue
                val = wr_matrix.loc[a, b] if (a in wr_matrix.index and b in wr_matrix.columns) else None
                if val is not None and not (isinstance(val, float) and pd.isna(val)):
                    if float(val) >= threshold:
                        tokens_to_keep.add(a)
                        tokens_to_keep.add(b)

        filtered_labels = [l for l in labels if l in tokens_to_keep]

        if not filtered_labels:
            st.info("Aucune paire ne dépasse ce seuil.")
        else:
            display_labels = [dn(l) for l in filtered_labels]
            z_vals, text_vals, hover_vals = [], [], []
            for a in filtered_labels:
                row_z, row_t, row_h = [], [], []
                for b in filtered_labels:
                    val = wr_matrix.loc[a, b] if (a in wr_matrix.index and b in wr_matrix.columns) else None
                    if a == b or val is None or (isinstance(val, float) and pd.isna(val)):
                        row_z.append(None)
                        row_t.append("")
                        row_h.append("—")
                    else:
                        v = float(val)
                        row_z.append(v if v >= threshold else None)
                        row_t.append(f"{v:.0%}" if v >= threshold else "")
                        row_h.append(f"{dn(a)} / {dn(b)}<br>Win rate : {v:.0%}")
                z_vals.append(row_z)
                text_vals.append(row_t)
                hover_vals.append(row_h)

            n = len(filtered_labels)
            cell_size = 44
            matrix_px = n * cell_size + 160

            fig_wr = go.Figure(go.Heatmap(
                z=z_vals, x=display_labels, y=display_labels,
                text=text_vals, hovertext=hover_vals,
                hovertemplate="%{hovertext}<extra></extra>",
                texttemplate="%{text}",
                colorscale=[
                    [0.0, "#fdf0f0"], [0.4, "#fef3e2"],
                    [0.6, "#e8f7f1"], [1.0, "#0F6E56"],
                ],
                zmin=0, zmax=1, showscale=False,
            ))
            # Shapes : lignes aux bordures des cellules pour un vrai quadrillage
            grid_shapes = []
            for i in range(n + 1):
                # lignes verticales
                grid_shapes.append(dict(
                    type="line", xref="x", yref="paper",
                    x0=i - 0.5, x1=i - 0.5, y0=0, y1=1,
                    line=dict(color="#ccc", width=1)
                ))
                # lignes horizontales
                grid_shapes.append(dict(
                    type="line", xref="paper", yref="y",
                    x0=0, x1=1, y0=i - 0.5, y1=i - 0.5,
                    line=dict(color="#ccc", width=1)
                ))

            fig_wr.update_layout(
                width=matrix_px,
                height=matrix_px,
                margin=dict(t=120, b=10, l=120, r=10),
                plot_bgcolor="#fff", paper_bgcolor="#fff",
                shapes=grid_shapes,
                xaxis=dict(tickfont=dict(size=10), side="top", showgrid=False, tickangle=-90),
                yaxis=dict(tickfont=dict(size=10), autorange="reversed", showgrid=False),
            )
            # Centrer via colonnes
            _, col_center, _ = st.columns([1, matrix_px // 10, 1])
            with col_center:
                st.plotly_chart(fig_wr, use_container_width=False)

with tab_logo:
    st.caption("Test de récupération des logos CoinMarketCap.")

    # Debug : afficher l'erreur si présente
    logos_data = st.session_state.get("token_logos", {})
    if "__error__" in logos_data:
        st.error(f"Erreur API : {logos_data['__error__']}")
    
    _key_present = "API_CMC" in st.secrets
    st.markdown(f"Clé API_CMC dans les secrets : **{'✅ trouvée' if _key_present else '❌ non trouvée'}**")
    st.markdown(f"Logos récupérés : **{sum(1 for k, v in logos_data.items() if not k.startswith('__') and isinstance(v, dict) and v.get('logo'))}** / {len(CRYPTOS)}")

    if st.button("🔄 Recharger les logos"):
        st.session_state["token_logos"] = {}
        st.rerun()

    rows_html = ""
    for name in CRYPTOS.keys():
        img   = logo_html(name, 20)
        dname = get_display_name(name)
        cmc   = TOKEN_CMC_MAP.get(name, "—")
        ok    = "✅" if get_logo(name) else "❌"
        rows_html += (
            f"<tr>"
            f"<td style='padding:4px 8px'>{img}</td>"
            f"<td style='padding:4px 8px;font-size:12px'>{dname}</td>"
            f"<td style='padding:4px 8px;font-size:11px;color:#888'>{cmc}</td>"
            f"<td style='padding:4px 8px'>{ok}</td>"
            f"</tr>"
        )
    st.markdown(
        f"<table style='border-collapse:collapse'><thead><tr>"
        f"<th style='padding:4px 8px;font-size:11px;color:#aaa;font-weight:400'>Logo</th>"
        f"<th style='padding:4px 8px;font-size:11px;color:#aaa;font-weight:400'>Token</th>"
        f"<th style='padding:4px 8px;font-size:11px;color:#aaa;font-weight:400'>Slug CMC</th>"
        f"<th style='padding:4px 8px;font-size:11px;color:#aaa;font-weight:400'></th>"
        f"</tr></thead><tbody>{rows_html}</tbody></table>",
        unsafe_allow_html=True
    )
