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
    """Scanne le dossier data/ et construit le dict {Nom: slug} depuis les noms de fichiers."""
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


METRICS_INFO = {
    "Corrélation": {
        "emoji": "📊",
        "definition": """Mesure si deux actifs bougent dans le même sens, en comparant leurs **rendements journaliers** (et non leurs prix bruts).

**Pourquoi les rendements et pas les prix ?**
Les prix montent structurellement avec le temps — comparer deux prix qui grimpent tous les deux donnerait une corrélation artificielle. Les rendements (variation % jour J vs jour J-1) éliminent cette tendance et mesurent uniquement le *comportement* commun.

**Formule :**
```
rendement_A(t) = (Prix_A(t) - Prix_A(t-1)) / Prix_A(t-1)

Corrélation = cov(rendements_A, rendements_B)
              ─────────────────────────────────
              std(rendements_A) × std(rendements_B)
```

**Lecture du résultat :**
- `1.0` → mouvements parfaitement identiques
- `0.0` → aucune relation
- `-1.0` → mouvements parfaitement opposés
- En pratique, une paire utile se situe entre **0.7 et 0.95**""",
        "seuil": "✅ Bon signal si > 0.7",
    },
    "Hedge Ratio (β)": {
        "emoji": "⚖️",
        "definition": """Répond à la question : *dans quelle proportion dois-je acheter B pour que ma position soit neutre par rapport au marché ?*

**Méthode : régression OLS (moindres carrés ordinaires)**
On cherche la droite qui minimise l'écart entre le prix de A et une combinaison linéaire de B.

**Formule :**
```
Prix_A = α + β × Prix_B + ε

β (hedge ratio) = cov(Prix_A, Prix_B) / var(Prix_B)
α (intercept)   = moyenne(Prix_A) - β × moyenne(Prix_B)
```

**Ce que ça donne concrètement :**
Si β = 0.003 entre Aave et Pendle, et que Pendle vaut 5$ et Aave vaut 90$ :
```
Capital A / Capital B = β × Prix_B / Prix_A
                      = 0.003 × 5 / 90 ≈ 0.00017
```
→ Pour 1 000$ de position, tu alloues ~999.83$ sur A et ~0.17$ sur B.

**Remarque :** un β très petit indique des prix très différents — c'est normal pour des cryptos de valeurs différentes.""",
        "seuil": "ℹ️ Pas de seuil fixe — dépend de la paire",
    },
    "Co-intégration (p)": {
        "emoji": "🔬",
        "definition": """La preuve mathématique qu'un **élastique invisible** relie les deux prix dans le temps.

**Deux actifs peuvent être corrélés sans être co-intégrés.** La corrélation mesure les mouvements simultanés. La co-intégration mesure si l'*écart* entre eux est stable sur le long terme — s'il revient toujours vers une moyenne.

**Calcul en 2 étapes :**

*Étape 1 — construire le spread :*
```
spread(t) = Prix_A(t) - (β × Prix_B(t) + α)
```
C'est l'écart entre le prix réel de A et ce que le modèle prédit.

*Étape 2 — test ADF (Augmented Dickey-Fuller) sur le spread :*
```
H0 (hypothèse nulle) : le spread est une marche aléatoire (pas de retour à la moyenne)
H1 (alternative)     : le spread est stationnaire (revient à sa moyenne)

p-value = probabilité d'observer ces données si H0 est vraie
```

**Lecture :**
- `p < 0.05` → on rejette H0 avec 95% de confiance → l'élastique existe ✅
- `p > 0.05` → on ne peut pas rejeter H0 → la relation est aléatoire ❌""",
        "seuil": "✅ Bon signal si p < 0.05",
    },
    "Half-Life (jours)": {
        "emoji": "⏳",
        "definition": """Le **chrono du trade** : combien de jours faut-il en moyenne pour que l'écart se réduise de moitié ?

**Modèle de retour à la moyenne (Ornstein-Uhlenbeck) :**
On suppose que le spread suit une dynamique où il est attiré vers sa moyenne avec une force proportionnelle à son éloignement.

**Formule :**
```
Δspread(t) = λ × spread(t-1) + ε

où λ est estimé par régression OLS :
  Δspread(t)   = spread(t) - spread(t-1)
  spread(t-1)  = valeur du spread la veille

Half-Life = ln(2) / λ
```

**Exemple :**
Si λ = 0.08 → Half-Life = ln(2) / 0.08 ≈ **8.7 jours**
→ Un trade ouvert aujourd'hui devrait se fermer en ~9 jours en moyenne.

**Pourquoi c'est important :**
- `< 3 jours` → trop court, probablement du bruit de marché
- `5–15 jours` → créneau idéal pour trader
- `> 30 jours` → capital immobilisé trop longtemps, risque de retournement""",
        "seuil": "✅ Idéal entre 5 et 15 jours",
    },
    "Z-Score": {
        "emoji": "🌡️",
        "definition": """Le **thermomètre de l'opportunité** : indique à quelle distance de sa moyenne historique se trouve le spread *en ce moment*, exprimé en nombre d'écarts-types.

**Formule (fenêtre glissante de 30 jours) :**
```
Z-Score(t) = spread(t) - moyenne_30j(spread)
             ──────────────────────────────────
             écart_type_30j(spread)
```

**Pourquoi une fenêtre de 30 jours ?**
On utilise une moyenne *mobile* plutôt qu'une moyenne fixe sur toute la période, car la relation entre deux cryptos évolue lentement. 30 jours capture le comportement récent sans sur-réagir au bruit quotidien.

**Signaux de trading :**
```
Z > +2  →  spread anormalement haut
           A est trop cher par rapport à B
           Signal : SHORT A + LONG B

Z < -2  →  spread anormalement bas
           A est trop bon marché par rapport à B
           Signal : LONG A + SHORT B

-2 < Z < +2  →  zone neutre, pas d'opportunité

Z = 0   →  spread exactement à sa moyenne, équilibre parfait
```

**Intuition :** si le z-score suit une distribution normale, un z > 2 se produit seulement ~2.5% du temps. C'est un événement statistiquement rare — c'est exactement là qu'on veut trader.""",
        "seuil": "🚨 Signal fort si |z| > 2",
    },
}

# ─── FONCTIONS ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_prices(slug):
    path = os.path.join(DATA_DIR, f"{slug}-historical-data.csv")
    if not os.path.exists(path):
        return None, f"Fichier introuvable : {slug}-historical-data.csv"
    try:
        df = pd.read_csv(path)

        # Nettoyage : retire $ et virgilles des colonnes numériques
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
st.caption(f"Données locales · {len(CRYPTOS)} tokens disponibles · dossier `data/`")

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
if "prefill_a" not in st.session_state:
    st.session_state.prefill_a = None
if "prefill_b" not in st.session_state:
    st.session_state.prefill_b = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0

tabs = st.tabs(["📚 Les 5 métriques", "🔍 Analyse d'une paire", "🗺️ Matrice des paires"])

# ══ TAB 1 — MÉTRIQUES ══════════════════════════════════════════════════════════

METRICS_COMPACT = {
    "Corrélation": {
        "emoji": "📊",
        "seuil": "> 0.7",
        "formule": "cov(rA, rB) / (σA × σB)",
        "note": "Calculée sur les rendements journaliers (pas les prix). Mesure si les deux actifs bougent dans le même sens.",
    },
    "Hedge Ratio β": {
        "emoji": "⚖️",
        "seuil": "Pas de seuil",
        "formule": "β = cov(A, B) / var(B)",
        "note": "Régression OLS de A sur B. Indique combien d'unités de B couvrent 1 unité de A.",
    },
    "Co-intégration p": {
        "emoji": "🔬",
        "seuil": "< 0.05",
        "formule": "ADF test sur spread = A − (β·B + α)",
        "note": "Test de stationnarité du spread. Si p < 0.05, l'écart entre les deux prix revient toujours à sa moyenne.",
    },
    "Half-Life": {
        "emoji": "⏳",
        "seuil": "5–15 jours",
        "formule": "ln(2) / λ  où  Δspread = λ · spread(t−1)",
        "note": "Modèle Ornstein-Uhlenbeck. Temps moyen pour que l'écart se réduise de moitié.",
    },
    "Z-Score": {
        "emoji": "🌡️",
        "seuil": "Signal si |z| > 2",
        "formule": "(spread − moy₃₀) / σ₃₀",
        "note": "Fenêtre glissante 30 jours. Un z > +2 se produit ~2.5% du temps — signal de trading.",
    },
}

with tabs[0]:
    st.divider()
    cols = st.columns(5)
    for col, (name, info) in zip(cols, METRICS_COMPACT.items()):
        with col:
            st.markdown(f"**{info['emoji']} {name}**")
            st.caption(f"Seuil : {info['seuil']}")
            st.code(info["formule"], language=None)
            st.markdown(f"<p style='font-size:11px;color:#888;line-height:1.5'>{info['note']}</p>", unsafe_allow_html=True)

# ══ TAB 2 — ANALYSE D'UNE PAIRE ════════════════════════════════════════════════
with tabs[1]:
    st.subheader("Analyse d'une paire")

    keys = list(CRYPTOS.keys())

    # Pré-remplissage depuis la matrice
    default_a = keys.index(st.session_state.prefill_a) if st.session_state.prefill_a in keys else min(0, len(keys)-1)
    default_b = keys.index(st.session_state.prefill_b) if st.session_state.prefill_b in keys else min(1, len(keys)-1)

    left, right = st.columns([1, 2], gap="large")

    with left:
        name_a = st.selectbox("Actif A", keys, index=default_a, key="sel_a")
        name_b = st.selectbox("Actif B", keys, index=default_b, key="sel_b")
        capital = st.number_input("Capital ($)", value=1000, step=100)
        analyse = st.button("Analyser", use_container_width=False)
        if name_a == name_b:
            st.warning("Choisis deux actifs différents.")

    with right:
        if name_a != name_b and analyse:
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
                    if m["verdict_color"] == "green":
                        st.success(f"**{m['Verdict']}** — co-intégration solide, half-life rapide.")
                    elif m["verdict_color"] == "orange":
                        st.warning(f"**{m['Verdict']}** — co-intégration ok mais paire lente.")
                    else:
                        st.error(f"**{m['Verdict']}** — co-intégration insuffisante.")

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

                    # ── BACKTEST ──────────────────────────────────────────────
                    st.divider()
                    st.markdown("#### Backtest")

                    spread = m["spread"]
                    z_score_series = m["z_score"].dropna()

                    entry_z = 2.0
                    exit_z = 0.0

                    trades = []
                    position = None
                    for date, z_val in z_score_series.items():
                        if position is None:
                            if z_val > entry_z:
                                position = {"type": "SHORT_A", "entry_z": z_val, "entry_date": date}
                            elif z_val < -entry_z:
                                position = {"type": "LONG_A", "entry_z": z_val, "entry_date": date}
                        else:
                            if abs(z_val) < exit_z or (position["type"] == "SHORT_A" and z_val < 0) or (position["type"] == "LONG_A" and z_val > 0):
                                pnl = abs(position["entry_z"]) - abs(z_val)
                                trades.append({
                                    "entrée": position["entry_date"].strftime("%Y-%m-%d"),
                                    "sortie": date.strftime("%Y-%m-%d"),
                                    "type": position["type"].replace("_", " "),
                                    "z entrée": round(position["entry_z"], 2),
                                    "z sortie": round(z_val, 2),
                                    "résultat": "✅ Gagnant" if pnl > 0 else "❌ Perdant",
                                })
                                position = None

                    if not trades:
                        st.info("Aucun trade déclenché sur la période avec ces paramètres.")
                    else:
                        df_trades = pd.DataFrame(trades)
                        n_trades = len(df_trades)
                        n_win = len(df_trades[df_trades["résultat"].str.contains("Gagnant")])
                        win_rate = n_win / n_trades if n_trades > 0 else 0

                        # P&L cumulé simulé sur le z-score
                        pnl_values = []
                        cum_pnl = 0
                        for _, t in df_trades.iterrows():
                            delta = abs(t["z entrée"]) - abs(t["z sortie"])
                            cum_pnl += delta * capital * 0.01
                            pnl_values.append(cum_pnl)

                        total_pnl = pnl_values[-1] if pnl_values else 0
                        max_dd = min(0, min(pnl_values)) if pnl_values else 0
                        returns = pd.Series(pnl_values).diff().dropna()
                        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

                        # Métriques backtest
                        b1, b2, b3, b4, b5 = st.columns(5)
                        b1.metric("P&L cumulé", f"{total_pnl:+.0f}$")
                        b2.metric("Trades", n_trades)
                        b3.metric("Win rate", f"{win_rate:.0%}")
                        b4.metric("Drawdown max", f"{max_dd:.0f}$")
                        b5.metric("Sharpe", f"{sharpe:.2f}")

                        # Graphique P&L
                        fig_pnl = go.Figure()
                        fig_pnl.add_trace(go.Scatter(
                            x=list(range(len(pnl_values))),
                            y=pnl_values,
                            mode="lines+markers",
                            line=dict(color="#1D9E75", width=1.5),
                            marker=dict(size=5),
                            name="P&L cumulé"
                        ))
                        fig_pnl.add_hline(y=0, line_dash="dot", line_color="rgba(150,150,150,0.5)", line_width=1)
                        fig_pnl.update_layout(
                            title=dict(text="P&L cumulé par trade", font=dict(size=12)),
                            height=200, margin=dict(t=36, b=16, l=40, r=16),
                            plot_bgcolor="#fff", paper_bgcolor="#fff",
                            showlegend=False,
                        )
                        fig_pnl.update_xaxes(title_text="Trade #", showgrid=False, tickfont=dict(size=10))
                        fig_pnl.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
                        st.plotly_chart(fig_pnl, use_container_width=True)

                        # Tableau des trades
                        with st.expander(f"Détail des {n_trades} trades"):
                            st.dataframe(df_trades, use_container_width=True, hide_index=True)

                    st.divider()

                    # Graphique prix normalisés
                    df = m["df"]
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
                        height=220, margin=dict(t=36, b=16, l=40, r=16),
                        plot_bgcolor="#fff", paper_bgcolor="#fff",
                        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0, font=dict(size=11)),
                    )
                    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
                    fig.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=10))
                    st.plotly_chart(fig, use_container_width=True)

                    # Graphique z-score
                    fig2 = go.Figure()
                    fig2.add_trace(go.Scatter(
                        x=z_score_series.index, y=z_score_series,
                        line=dict(color="#378ADD", width=1.5),
                        fill="tozeroy", fillcolor="rgba(55,138,221,0.05)"
                    ))
                    for y_val, color in [(2, "rgba(220,50,50,0.5)"), (-2, "rgba(220,50,50,0.5)"), (0, "rgba(180,180,180,0.5)")]:
                        fig2.add_hline(y=y_val, line_dash="dash", line_color=color, line_width=1)
                    fig2.update_layout(
                        title=dict(text="Z-Score — signal de trading", font=dict(size=12)),
                        height=220, margin=dict(t=36, b=16, l=40, r=16),
                        plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
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

        # Lecture locale — pas de rate limit, pas de délai
        cache = {}
        errors = []
        for i, name in enumerate(selected):
            cache[name], err = fetch_prices(CRYPTOS[name])
            if err:
                errors.append(f"{name} : {err}")
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
                "Beta (β)": m["Hedge Ratio (β)"],
                "p-value": m["Co-intégration (p)"],
                "Half-Life (j)": m["Half-Life (jours)"],
                "Z-Score": m["Z-Score"],
                "Verdict": m["Verdict"],
                "Signal": m["Signal"],
            })

        progress.empty()

        if not results:
            st.error("Aucun résultat. Vérifie que tes fichiers CSV sont bien dans le dossier data/.")
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

            # Tableau — paires avec signal actif
            st.markdown("#### Paires avec signal actif")
            df_signal = df_res[df_res["Signal"] != "Pas de signal"].copy()

            # Tableau — paires avec signal actif ET co-intégration valide
            st.markdown("#### Paires avec signal actif")
            df_signal = df_res[
                (df_res["Signal"] != "Pas de signal") &
                (df_res["Verdict"] != "❌ Faible")
            ].copy()

            if df_signal.empty:
                st.info("Aucune opportunité valide en ce moment — soit pas de signal z-score, soit co-intégration insuffisante.")
            else:
                # En-tête
                h1, h2, h3, h4, h5, h6, h7 = st.columns([2.2, 1, 1, 1, 1, 1, 1.8])
                for h, label in zip(
                    [h1, h2, h3, h4, h5, h6, h7],
                    ["Paire", "Corrélation", "Beta (β)", "p-value", "Half-Life", "Z-Score", "Signal"]
                ):
                    h.markdown(f"<p style='font-size:11px;color:#999;margin:0'>{label}</p>", unsafe_allow_html=True)
                st.divider()

                for idx, row in df_signal.iterrows():
                    c1, c2, c3, c4, c5, c6, c7 = st.columns([2.2, 1, 1, 1, 1, 1, 1.8])
                    verdict_icon = "✅" if "✅" in row["Verdict"] else "⚠️"
                    z_color = "#E24B4A" if abs(float(row["Z-Score"])) > 2 else "inherit"
                    p_color = "#1D9E75" if float(row["p-value"]) < 0.05 else "#E24B4A"

                    c1.markdown(f"**{row['Paire']}** {verdict_icon}")
                    c2.markdown(f"<span style='font-size:12px'>{row['Corrélation']:.3f}</span>", unsafe_allow_html=True)
                    c3.markdown(f"<span style='font-size:12px'>{row['Beta (β)']:.4f}</span>", unsafe_allow_html=True)
                    c4.markdown(f"<span style='font-size:12px;color:{p_color}'>{row['p-value']:.4f}</span>", unsafe_allow_html=True)
                    c5.markdown(f"<span style='font-size:12px'>{row['Half-Life (j)']} j</span>", unsafe_allow_html=True)
                    c6.markdown(f"<span style='font-size:12px;font-weight:500;color:{z_color}'>{row['Z-Score']:.2f}</span>", unsafe_allow_html=True)
                    c7.markdown(f"<span style='font-size:11px'>{row['Signal']}</span>", unsafe_allow_html=True)

                    # Graphe détail pleine largeur sous la ligne
                    pair_a, pair_b = row["Paire"].split(" / ")
                    with st.expander(f"Détail z-score — {row['Paire']}", expanded=False):
                        sa, ea = fetch_prices(CRYPTOS[pair_a])
                        sb, eb = fetch_prices(CRYPTOS[pair_b])
                        if ea or eb:
                            st.error(ea or eb)
                        else:
                            md = compute_metrics(sa, sb, pair_a, pair_b)
                            if md:
                                z_s = md["z_score"].dropna()
                                df_ab = md["df"]

                                col_g1, col_g2 = st.columns(2)
                                with col_g1:
                                    fig_z = go.Figure()
                                    fig_z.add_trace(go.Scatter(
                                        x=z_s.index, y=z_s,
                                        line=dict(color="#378ADD", width=1.2),
                                        fill="tozeroy", fillcolor="rgba(55,138,221,0.05)"
                                    ))
                                    for yv, col in [(2, "rgba(220,50,50,0.4)"), (-2, "rgba(220,50,50,0.4)"), (0, "rgba(180,180,180,0.4)")]:
                                        fig_z.add_hline(y=yv, line_dash="dash", line_color=col, line_width=0.8)
                                    fig_z.update_layout(
                                        title=dict(text="Z-Score", font=dict(size=11)),
                                        height=200, margin=dict(t=30, b=10, l=30, r=10),
                                        plot_bgcolor="#fff", paper_bgcolor="#fff", showlegend=False,
                                    )
                                    fig_z.update_xaxes(showgrid=False, tickfont=dict(size=9))
                                    fig_z.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=9))
                                    st.plotly_chart(fig_z, use_container_width=True)

                                with col_g2:
                                    fig_p = go.Figure()
                                    fig_p.add_trace(go.Scatter(
                                        x=df_ab.index, y=df_ab["A"] / df_ab["A"].iloc[0],
                                        name=pair_a, line=dict(color="#1D9E75", width=1.2)
                                    ))
                                    fig_p.add_trace(go.Scatter(
                                        x=df_ab.index, y=df_ab["B"] / df_ab["B"].iloc[0],
                                        name=pair_b, line=dict(color="#7F77DD", width=1.2)
                                    ))
                                    fig_p.update_layout(
                                        title=dict(text="Prix normalisés", font=dict(size=11)),
                                        height=200, margin=dict(t=30, b=10, l=30, r=10),
                                        plot_bgcolor="#fff", paper_bgcolor="#fff",
                                        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
                                    )
                                    fig_p.update_xaxes(showgrid=False, tickfont=dict(size=9))
                                    fig_p.update_yaxes(showgrid=True, gridcolor="#f0ede6", tickfont=dict(size=9))
                                    st.plotly_chart(fig_p, use_container_width=True)

            # Tableau complet (toutes les paires)
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

            styled_all = (
                df_res.style
                .applymap(color_verdict, subset=["Verdict"])
                .applymap(color_pvalue, subset=["p-value"])
                .applymap(color_zscore, subset=["Z-Score"])
                .format({
                    "Corrélation": "{:.3f}",
                    "Beta (β)": "{:.4f}",
                    "p-value": "{:.4f}",
                    "Z-Score": "{:.2f}",
                })
            )
            st.dataframe(styled_all, use_container_width=True, height=480)

            # Top 5
            st.markdown("#### 🏆 Meilleures configurations")
            ideal = df_res[df_res["Verdict"] == "✅ Idéale"].sort_values("p-value")
            if len(ideal) == 0:
                st.info("Aucune paire idéale sur cette période. Consulte le tableau.")
            else:
                for _, row in ideal.head(5).iterrows():
                    signal_txt = f" · **{row['Signal']}**" if row["Signal"] != "Pas de signal" else ""
                    st.success(
                        f"**{row['Paire']}** · "
                        f"corr {row['Corrélation']} · "
                        f"p {row['p-value']} · "
                        f"hl {row['Half-Life (j)']} j · "
                        f"z {row['Z-Score']}"
                        f"{signal_txt}"
                    )
