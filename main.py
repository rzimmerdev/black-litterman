import numpy as np
import pandas as pd
import dxlib as dx
from dxlib.interfaces.external import yfinance
import matplotlib.pyplot as plt

from fixed_income import Futures, IPCA
from utils import dirichlet_weights, black_litterman, mean_variance_optim

tickers = {
    'Treasury': 'ZN=F',  # 10-Year T-Note Future 1st contract
    'S&P500': 'ES=F',  # E-mini S&P 500 Future 1st contract
    'DOL-1stFuture': 'BRL=X',  # USD/BRL -> US Dollar to Brazilian Real
    'Euro-1stFuture': 'EURBRL=X',  # EUR/BRL -> Euro to Brazilian Real
}


def build_portfolio(history: dx.History):
    """
    Expected tickers:

    ZN=F
    ES=F
    BRL=X
    EURBRL=X
    DI1F26
    DI1F29
    IND1!
    CNYBRL=X
    IPCA2029
    IPCA2050

    Args:
        history (DataFrame): DataFrame with the historical data of the securities

    Returns:
        DataFrame: DataFrame with the portfolio weights
    """
    securities = ['ZN=F', 'ES=F', 'BRL=X', 'EURBRL=X', 'DI1F26', 'DI1F29', 'IND1!', 'CNYBRL=X', 'IPCA2029', 'IPCA2050']
    gamma = 1
    tau = 1

    pick_matrix = pd.DataFrame(np.zeros((len(securities), len(securities))), columns=securities)
    views = pd.Series(np.zeros(len(securities)), index=securities)

    view_uncertainty = pd.Series(np.ones(len(securities)), index=securities)
    # diagonalize
    view_uncertainty = pd.DataFrame(np.diag(view_uncertainty), columns=securities, index=securities)

    def log_returns(x):
        return np.log(x).diff()

    equilibrium_returns = history.apply_df({dx.SchemaLevel.SECURITY: lambda x: x.pct_change().mean()})
    equilibrium_returns = equilibrium_returns['close']

    df = history.df.copy()
    # index level 0 = date
    # index level 1 = security
    # column 1 = close, with close values
    # new df:
    # index = date
    # columns = securities, with close values
    df = df.reset_index()
    # drop hours from date
    df['date'] = df['date'].dt.date
    df_agg = df.groupby(['date', 'security'], as_index=False).agg({'close': 'mean'})
    df = df_agg.pivot(index='date', columns='security', values='close')

    # drop weekends
    df.index = pd.to_datetime(df.index)
    df = df[df.index.dayofweek < 5]
    # forward fill
    df = df.fillna(method='ffill')
    cov = df.apply(log_returns).dropna().cov()

    returns_bl = black_litterman(pick_matrix, views, view_uncertainty, equilibrium_returns, cov, tau)
    alpha = gamma * np.ones(len(securities))
    weights = dirichlet_weights(returns_bl, cov, alpha)
    return weights


def main():
    api = yfinance.API()
    histories = api.historical(list(tickers.values()), '2023-07-17', '2024-07-16', cache=True).get(fields=['close'])
    histories.schema.fields = ["close"]
    schema = histories.schema

    # also add from csv
    # use only Fechamento column, rename Fechamento column to close
    futures = Futures().preprocess(schema)
    ipca = IPCA().preprocess(schema)

    history = histories
    for future in futures.values():
        history = history + future
    for fixed in ipca.values():
        history = history + fixed

    for security in history.schema.security_manager:
        print(security)

    weights = build_portfolio(history)

    print(weights)

    # Add together percentages that are less than 0.1% into a 'Outros' and plot in separate pie
    weights = weights[weights > 1e-4]
    sep = 0.05
    others = weights[weights < sep]

    if others.size > 0:
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        weights = weights[weights >= sep]
        weights['Outros'] = others.sum()
        axs[0].pie(weights, labels=weights.index, autopct='%1.1f%%')
        axs[0].set_title('Pesos da carteira')
        axs[1].pie(others, labels=others.index, autopct='%1.1f%%')
        axs[1].set_title(f"Pesos da carteira - Outros ({others.sum():.2f}%)")
        plt.show()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        axs.pie(weights, labels=weights.index, autopct='%1.1f%%')
        axs.set_title('Pesos da carteira')
        plt.show()


if __name__ == "__main__":
    main()
