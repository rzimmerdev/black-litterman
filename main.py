from typing import List, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
import dxlib as dx
from dxlib.interfaces.external import yfinance
import matplotlib.pyplot as plt
from scipy.stats import norm

from fixed_income import Futures, IPCA
from utils import dirichlet_weights, black_litterman

available_tickers = {
    'Treasury': 'ZN=F',  # 10-Year T-Note Future 1st contract
    'S&P500': 'ES=F',  # E-mini S&P 500 Future 1st contract
    'USD/BRL': 'BRL=X',  # USD/BRL -> US Dollar to Brazilian Real
    'EUR/BRL': 'EURBRL=X',  # EUR/BRL -> Euro to Brazilian Real
}

ibrx_tickers = pd.read_csv('ibrx100.csv').values.reshape(-1).tolist()
bdr_tickers = pd.read_csv('bdrs.csv').values.reshape(-1).tolist()


def load_history_groups():
    futures = Futures().preprocess()
    ipca = IPCA().preprocess()

    api = yfinance.API()
    external_tickers = list(available_tickers.values())
    api_fx = api.historical(external_tickers, '2023-07-17', '2024-07-16', cache=True).get(fields=['close'])
    api_bdr = api.historical(bdr_tickers, '2023-07-17', '2024-07-16', cache=True).get(fields=['close'])
    api_ibrx = api.historical(ibrx_tickers, '2023-07-17', '2024-07-16', cache=True).get(fields=['close'])
    sm = api_fx.schema.security_manager

    ipca_pre = ipca['IPCA2029'] + ipca['IPCA2050'] + futures['DI1F26'] + futures['DI1F29']
    dollar = api_fx.get(levels={
        dx.SchemaLevel.SECURITY: [sm.get('BRL=X')]
    })
    yuan = futures['CNYBRL=X']
    euro = api_fx.get(levels={
        dx.SchemaLevel.SECURITY: [sm.get('EURBRL=X')]
    })
    treasury = api_fx.get(levels={
        dx.SchemaLevel.SECURITY: [sm.get('ZN=F')]
    })
    brazilian = futures['IND1!'] + api_ibrx
    bdr = api_bdr
    spy = api_fx.get(levels={
        dx.SchemaLevel.SECURITY: [sm.get('ES=F')]
    })

    histories = {
        'ipca': ipca_pre,
        'dollar': dollar,
        'yuan': yuan,
        'euro': euro,
        'treasury': treasury,
        'brazilian': brazilian,
        'bdr': bdr,
        'spy': spy
    }

    return histories


def create_convex_variables(name, history,
                            upper_limit=1,
                            lower_limit=0,
                            max_quantity=None) -> Tuple[cp.Variable, cp.Variable, List[cp.Constraint]]:
    """
    Creates and returns inputs to a CVXPY problem, specifically:
    - the adjustable variables (weights)
    - the constraints, given input arguments

    Args:
        name (str): The name of the variable
        history (dx.History): The historical data
        upper_limit (float): The maximum portfolio weight for this group of assets
        lower_limit (float): The minimum portfolio weight for this group of assets
        max_quantity (int): The maximum number of assets to be selected within the group.
                            The default is None, which makes the number of assets unrestricted.

    Returns:
        tuple: A tuple containing the adjustable variables and the list of constraints
    """
    securities = history.df.reset_index()['security'].unique()
    n = len(securities)

    # These are the weights of the portfolio
    w = cp.Variable(n, name=f"{name}_weights")
    # These are the binary variables for the stocks (1 if selected, 0 otherwise)
    y = cp.Variable(n, boolean=True, name=f"{name}_selected")

    # Create the constraints
    constraints = [
        # cp.sum(w) == 1, no need to enforce this constraint, since the current group can be a subset of the portfolio
        w >= lower_limit * y,
        w <= upper_limit * y,
        cp.sum(w) <= upper_limit,
        cp.sum(w) >= lower_limit,
        cp.sum(y) <= (max_quantity if max_quantity is not None else n),
    ]

    return w, y, constraints


def build_portfolio(histories: dict, bounds: dict) -> Tuple[pd.Series, pd.DataFrame]:
    # First step is to sum up histories so as to calculate the entire covariance matrix, not just intergroup covariances
    df = pd.concat([history.df for history in histories.values()])
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

    def log_returns(x):
        return np.log(x).diff()

    returns = df.apply(log_returns).dropna()
    cov = returns.cov()

    weights: dict[str, cp.Variable] = {}
    selected: dict[str, cp.Variable] = {}
    constraints: dict[str, List[cp.Constraint]] = {}
    for group in histories:
        history = histories[group]
        # sub covariance matrix, only for the group
        weight, select, constraint = create_convex_variables(group, history, **(bounds[group]))
        weights[group] = weight
        selected[group] = select
        constraints[group] = constraint

    final_weights = cp.hstack([w for w in weights.values()])
    final_selected = cp.hstack([y for y in selected.values()])

    # additional VaR constraint
    conf = 0.99
    bound = 3e-2
    z_score = norm.ppf(conf)
    std = cp.sqrt(cp.quad_form(final_weights, cov))
    var = cp.matmul(returns.values, final_weights) - z_score * std
    # constraints['var'] = [var >= bound] -> doesn't work, need to use a list comprehension
    # constraints['var'] = [cp.sum(var) >= bound]

    # additional constraint to ensure the sum of absolute weights
    constraints['sum'] = [cp.sum(cp.abs(final_weights)) <= 1]

    gamma = 1e-2
    tau = 2
    securities = list(returns.columns)
    sm = dx.SecurityManager.from_list(securities)
    view_names = ["Hedge BRL/USD with CNY/BRL"]
    num_views = len(view_names)

    pick_matrix = pd.DataFrame(np.zeros((num_views, len(securities))), columns=securities)
    views = pd.Series(np.zeros(num_views), index=view_names)

    view_uncertainty = pd.Series(np.ones(num_views), index=view_names)
    view_uncertainty = pd.DataFrame(np.diag(view_uncertainty), index=view_names, columns=view_names)

    def set_pick(ticker, value):
        security = sm.get(ticker)
        pick_matrix.loc[0, security] = value

    def set_view(name, value, uncertainty):
        views.loc[name] = value
        view_uncertainty.loc[name, name] = uncertainty

    # hedge BRL/USD with CNY/BRL
    # set_pick('BRL=X', 1)
    # set_pick('CNYBRL=X', -1)
    # set_view("Hedge BRL/USD with CNY/BRL", 0.01, 0.05)

    # print views, and their affected securities
    for i, row in pick_matrix.iterrows():
        view = views[i]
        affected_securities = row[row != 0].index
        print(f"View {i}: {view} on {affected_securities}")

    mean_returns = returns.mean()
    returns_bl = black_litterman(pick_matrix, views, view_uncertainty, mean_returns, cov, tau)

    # problem definition -> mean-variance optimization with cardinality constraint
    def optim(mu, sigma):
        # Objective function: Mean-variance optimization
        # Min(0.5 * cp.quad_form(w, sigma) - gamma * mu @ w))
        objective = cp.Minimize(
            0.5 * cp.quad_form(final_weights, sigma) - gamma * cp.matmul(mu, final_weights)
        )

        problem = cp.Problem(objective, [c for group in constraints.values() for c in group])
        problem.solve(solver=cp.ECOS_BB)

        return final_weights.value * final_selected.value

    alpha = np.ones(num_views)
    return dirichlet_weights(returns_bl, cov, alpha, optim), df


def main():
    histories = load_history_groups()

    history = histories['ipca']
    for group in histories:
        history = history + histories[group]

    bounds = {
        'ipca': {'upper_limit': 2},
        'dollar': {'upper_limit': .3, 'lower_limit': -.3},
        'yuan': {'upper_limit': .3, 'lower_limit': -.3},
        'euro': {'upper_limit': .3, 'lower_limit': -.3},
        'treasury': {'upper_limit': 1, 'lower_limit': -.1},
        'brazilian': {'upper_limit': .25, 'max_quantity': 7},
        'bdr': {'upper_limit': .2, 'max_quantity': 4},
        'spy': {'upper_limit': .2, 'lower_limit': .1}
    }

    securities = history.df.reset_index()['security'].unique()
    optim_weights, df = build_portfolio(histories, bounds)
    optim_weights = pd.Series(optim_weights, index=securities)
    optim_weights = optim_weights[optim_weights.abs() > 1e-4]

    # Add together percentages that are less than 0.01% into a 'Outros' and plot in separate pie
    selected_securities = optim_weights.index
    optim_weights.index = [f"-{security.ticker}" if weight < 0 else security.ticker
                           for security, weight in optim_weights.items()]
    total_pct = optim_weights.sum() * 100
    weights = optim_weights.abs()

    sep = 0.05
    if weights[weights < sep].size > 0 and sum(weights[weights < sep]) > 0.09:
        others = weights[weights < sep]
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        weights = weights[weights >= sep]
        weights['Outros'] = others.sum()
        axs[0].pie(weights, autopct='%1.1f%%', labels=weights.index)
        axs[0].set_title(f"Pesos da carteira - {total_pct:.2f}%")
        axs[1].pie(others, autopct='%1.1f%%', labels=others.index)
        axs[1].set_title(f"Pesos da carteira - Outros ({others.sum():.2f}%)")
        plt.show()
    else:
        fig, axs = plt.subplots(1, 1, figsize=(15, 5))
        axs.pie(weights, autopct='%1.1f%%', labels=weights.index)
        axs.set_title(f"Pesos da carteira - {total_pct:.2f}%")
        plt.show()

    # backtest of the portfolio
    # get the returns of the selected securities
    returns = df[selected_securities].pct_change().dropna()

    # plot accumulated returns graph, and the portfolio accumulated returns, on the same graph
    acc_portfolio = (1 + returns @ weights.values).cumprod()
    acc_returns = (1 + returns).cumprod()
    acc_returns['Portfolio'] = acc_portfolio

    for column in acc_returns.columns:
        if column == 'Portfolio':
            acc_returns[column].plot(figsize=(15, 5), color='cyan', linestyle='--', label='Portfolio')
        else:
            acc_returns[column].plot(alpha=0.65)

    plt.title('Retornos acumulados')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
