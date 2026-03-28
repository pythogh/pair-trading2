import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def analyze_pair(ticker_a, ticker_b, capital_total=1000):
    print(f"\n" + "="*60)
    print(f"🚀 RAPPORT D'ANALYSE STATISTIQUE : {ticker_a} / {ticker_b}")
    print("="*60)

    # 1. Téléchargement des données (6 mois)
    data_a = yf.download(ticker_a, period="6mo", interval="1d", progress=False)['Close']
    data_b = yf.download(ticker_b, period="6mo", interval="1d", progress=False)['Close']

    df = pd.concat([data_a, data_b], axis=1).dropna()
    df.columns = ['Price_A', 'Price_B']

    # --- CALCULS ---
    # CORRÉLATION
    returns = df.pct_change().dropna()
    correlation = returns['Price_A'].corr(returns['Price_B'])

    # BETA (Hedge Ratio)
    X = sm.add_constant(df['Price_B'])
    model = sm.OLS(df['Price_A'], X).fit()
    beta = model.params['Price_B']
    alpha = model.params['const']

    # CO-INTÉGRATION
    spread = df['Price_A'] - (beta * df['Price_B'] + alpha)
    p_value = adfuller(spread.dropna())[1]

    # Z-SCORE
    z_score = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

    # HALF-LIFE
    spread_lag = spread.shift(1)
    spread_diff = spread.diff()
    valid = ~(spread_lag.isna() | spread_diff.isna())
    res = sm.OLS(spread_diff[valid], spread_lag[valid]).fit()
    lambda_val = -res.params.iloc[0]
    half_life = np.log(2) / lambda_val if lambda_val > 0 else float('inf')

    # --- AFFICHAGE AVEC EXPLICATIONS ---
    print(f"📊 CORRÉLATION : {correlation:.2f}")
    print(f"   -> Mesure si les deux actifs bougent ensemble. Plus on est proche de 1.00, plus la paire est 'siamaoise'.")

    print(f"\n⚖️  HEDGE RATIO (BETA) : {beta:.4f}")
    print(f"   -> C'est le réglage de l'équilibre. Il indique qu'il faut environ {beta:.2f} unités de {ticker_b} pour compenser 1 unité de {ticker_a}.")

    print(f"\n🔬 CO-INTÉGRATION : p={p_value:.4f}")
    print(f"   -> C'est la preuve de l'existence d'un 'élastique' entre les prix. Si p < 0.05, l'écart finit toujours par revenir à sa moyenne.")

    print(f"\n⏳ HALF-LIFE : {half_life:.1f} jours")
    print(f"   -> C'est le chrono du trade. C'est le temps moyen estimé pour que l'écart (spread) se réduise de moitié et revienne vers l'équilibre.")

    print(f"\n🌡️  Z-SCORE ACTUEL : {z_score.iloc[-1]:.2f}")
    print(f"   -> C'est le thermomètre de l'opportunité. Un score de ±2 signifie que l'élastique est extrêmement tendu et prêt à lâcher.")

    print("-" * 60)

    # --- LOGIQUE DE TRADING ET ALLOCATION ---
    actual_z = z_score.iloc[-1]
    price_a_now = df['Price_A'].iloc[-1]
    price_b_now = df['Price_B'].iloc[-1]

    # Calcul pour la neutralité Beta (Market Neutral)
    ratio_capital = abs(beta * price_b_now / price_a_now)
    alloc_a = capital_total / (1 + ratio_capital)
    alloc_b = capital_total - alloc_a

    if p_value < 0.05 and half_life < 15:
        print("✅ VERDICT : Configuration IDÉALE. La relation mathématique est solide et rapide.")
    else:
        print("⚠️ VERDICT : Prudence. La paire manque de co-intégration ou est trop lente à réagir.")

    if actual_z > 2:
        print(f"\n🚨 SIGNAL : SHORT {ticker_a} / LONG {ticker_b}")
        print(f"   L'écart est trop haut : on vend l'actif cher et on achète le moins cher.")
        print(f"   👉 ACTION : Vendre {alloc_a:.2f}$ de {ticker_a} | Acheter {alloc_b:.2f}$ de {ticker_b}")
    elif actual_z < -2:
        print(f"\n🚨 SIGNAL : LONG {ticker_a} / SHORT {ticker_b}")
        print(f"   L'écart est trop bas : on achète l'actif bradé et on vend celui qui surperforme.")
        print(f"   👉 ACTION : Acheter {alloc_a:.2f}$ de {ticker_a} | Vendre {alloc_b:.2f}$ de {ticker_b}")
    else:
        print("\n😴 ÉTAT : Équilibre maintenu. Le marché est 'juste', aucune opportunité statistique ici.")

    print("="*60)

# --- CONFIGURATION ---
MON_CAPITAL = 1000 
analyze_pair('AAVE-USD', 'PENDLE-USD', capital_total=MON_CAPITAL)
