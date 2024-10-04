from typing import List, Tuple

import numpy as np
import pandas as pd
import cvxpy as cp
import dxlib as dx
import matplotlib.pyplot as plt
import locale
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator, FuncFormatter
from dxlib.interfaces.external import yfinance

from fixed_income import Futures, IPCA
from utils import black_litterman

available_tickers = {
    'Treasury': 'ZN=F',  # 10-Year T-Note Future 1st contract
    'S&P500': 'ES=F',  # E-mini S&P 500 Future 1st contract
    'USD/BRL': 'BRL=X',  # USD/BRL -> US Dollar to Brazilian Real
    'EUR/BRL': 'EURBRL=X',  # EUR/BRL -> Euro to Brazilian Real
}

ibrx = pd.read_csv('ibrx100.csv')
bdr = pd.read_csv('bdrs.csv')
ibrx_tickers = ibrx['tickers'].values.reshape(-1).tolist()
bdr_tickers = bdr['tickers'].values.reshape(-1).tolist()

all_stocks = pd.concat([ibrx, bdr], ignore_index=True)


def load_history_groups():
    futures = Futures().preprocess()
    ipca = IPCA().preprocess()

    api = yfinance.MarketInterface()
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


def create_convex_variables(name, securities,
                            upper_limit=1,
                            lower_limit=0,
                            max_quantity=None) -> Tuple[cp.Variable, cp.Variable, List[cp.Constraint]]:
    """
    Creates and returns inputs to a CVXPY problem, specifically:
    - the adjustable variables (weights)
    - the constraints, given input arguments

    Args:
        name (str): The name of the variable
        securities (dx.History): The historical data
        upper_limit (float): The maximum portfolio weight for this group of assets
        lower_limit (float): The minimum portfolio weight for this group of assets
        max_quantity (int): The maximum number of assets to be selected within the group.
                            The default is None, which makes the number of assets unrestricted.

    Returns:
        tuple: A tuple containing the adjustable variables and the list of constraints
    """
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
    tau = 1
    var_bound = 1e-2
    max_sum = 1.5

    # First step is to sum up histories so as to calculate the entire covariance matrix, not just intergroup covariances
    df = pd.concat([history.df for history in histories.values()]).reset_index()
    # group by same date and security and take the mean
    df_agg = df.groupby([dx.SchemaLevel.DATE, dx.SchemaLevel.SECURITY], sort=False).mean().reset_index()
    df = df_agg.pivot(index=dx.SchemaLevel.DATE, columns=dx.SchemaLevel.SECURITY, values='close')

    # drop weekends
    df.index = pd.to_datetime(df.index)
    df = df[df.index.dayofweek < 5]
    # forward fill
    df = df.fillna(method='ffill')

    def log_returns(x):
        return np.log(x).diff()

    returns = df.apply(log_returns).dropna()
    cov = returns.cov()

    securities = list(returns.columns)
    view_names = ["Hedge BRL/USD with CNY/BRL", "Positive view for SPY"]
    num_views = len(view_names)

    pick_matrix = pd.DataFrame(np.zeros((num_views, len(securities))), columns=securities, index=view_names)
    views = pd.Series(np.zeros(num_views), index=view_names)

    view_uncertainty = pd.Series(np.ones(num_views), index=view_names)
    view_uncertainty = pd.DataFrame(np.diag(view_uncertainty), index=view_names, columns=view_names)

    sm = dx.SecurityManager.from_list(securities)

    def set_pick(name, ticker, value, oppose=False):
        if not oppose:
            security = sm.get(ticker)
            pick_matrix.loc[name, security] = value
        else:
            # make all that are not the security have the opposite value
            for security in securities:
                if security.ticker != ticker:
                    pick_matrix.loc[name, security] = -value

    def set_view(name, value, uncertainty):
        views.loc[name] = value
        view_uncertainty.loc[name, name] = uncertainty

    def in_sector(sector: str):
        # count how many securities in all_stocks are in the sector
        return all_stocks[all_stocks['sector'] == sector].shape[0]

    def set_sector(name, sector, value):
        # set the pick for all securities in the sector
        for security in all_stocks[all_stocks['sector'] == sector]['tickers']:
            set_pick(name, security, value)

    # hedge BRL/USD with CNY/BRL
    set_view("Hedge BRL/USD with CNY/BRL", 0.01, 0.05)
    set_pick("Hedge BRL/USD with CNY/BRL", 'BRL=X', 1)
    set_pick("Hedge BRL/USD with CNY/BRL", 'CNYBRL=X', -1)

    # positive view for SPY -> We believe SPY will outperform the market
    # set_pick("Positive view for SPY", 'ES=F', 1)
    # set_pick("Positive view for SPY", 'ES=F', 1/(len(securities) - 1), oppose=True)
    # set_view("Positive view for SPY", -0.15, 0.05)

    # negative view for DI1F26 and DI1F29 -> We believe rates will go up
    # set_pick("Negative view for DI1F26 and DI1F29", 'DI1F26', -1)
    # set_pick("Negative view for DI1F26 and DI1F29", 'DI1F29', -1)
    # set_pick("Negative view for DI1F26 and DI1F29", 'DI1F26', -1/(len(securities) - 2), oppose=True)
    # set_view("Negative view for DI1F26 and DI1F29", 0.02, 0.05)

    # print views, and their affected securities
    for i, row in pick_matrix.iterrows():
        view = views[i]
        affected_securities = row[row != 0].index
        print(f"View {i}: {view} on {affected_securities}")

    returns = returns.mean()

    # problem definition -> mean-variance optimization with cardinality constraint
    weights: dict[str, cp.Variable] = {}
    selected: dict[str, cp.Variable] = {}
    constraints: dict[str, List[cp.Constraint]] = {}

    for group in histories:
        history = histories[group]

        # if empty, skip
        if history.df.empty:
            continue
        securities = history.df.reset_index()[dx.SchemaLevel.SECURITY].unique()
        # sub covariance matrix, only for the group
        weight, select, constraint = create_convex_variables(group, securities, **(bounds[group]))
        weights[group] = weight
        selected[group] = select
        constraints[group] = constraint

    final_weights = cp.hstack([w for w in weights.values()])
    final_selected = cp.hstack([y for y in selected.values()])

    # additional constraint to ensure the sum of absolute weights
    constraints['sum'] = [cp.sum(cp.abs(final_weights)) <= max_sum]

    # VaR constraint for entire portfolio
    portfolio_variance = cp.quad_form(final_weights, cov)

    # Add constraints to ensure the portfolio variance is close to desired risk
    # Cant use cp.sqrt since is not DCP compliant
    # noinspection PyTypeChecker
    constraints['risk'] = [cp.abs(portfolio_variance) <= var_bound ** 2]

    objective = cp.Maximize(cp.matmul(returns, final_weights))

    problem = cp.Problem(objective, [c for group in constraints.values() for c in group])
    problem.solve(solver=cp.ECOS_BB)

    # use weights per group to keep original names and indices order
    final_weights = pd.Series()

    for group, weight in weights.items():
        securities = histories[group].df.reset_index()[dx.SchemaLevel.SECURITY].unique()
        final_weights = pd.concat([final_weights, pd.Series(weight.value, index=securities)])

    return final_weights, df


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

    optim_weights, df = build_portfolio(histories, bounds)
    optim_weights = pd.Series(optim_weights, index=df.columns)

    if use_fictional := True:
        sm = dx.SecurityManager.from_list(df.columns)

        def get(ticker):
            security = sm.get(ticker)
            if security is None:
                sm.add(ticker)
                security = sm.get(ticker)
            return security

        optim_weights = pd.Series({get(key): value / 100 for key, value in {
            "IPCA2029": 6.1,
            "IPCA2050": 6.6,
            "DI1F26": 15.14,
            "CNYBRL=X": -8.89,
            "BRL=X": -10.66,
            "WEGE3.SA": 7.50,
            "BBAS3.SA": 6.2,
            "ITUB4.SA": 3.11,
            "GGBR4.SA": 6.37,
            "ES=F": 20.0,
            "NVDC34.SA": 5.12,
            "MSFT34.SA": 4.31,
        }.items()})

    optim_weights = optim_weights[optim_weights.abs() > 1e-4]

    # Add together percentages that are less than 0.01% into a 'Outros' and plot in separate pie
    selected_securities = optim_weights.index
    optim_weights.index = [f"-{security.ticker}" if weight < 0 else security.ticker
                           for security, weight in optim_weights.items()]
    weights = optim_weights.abs()
    total_pct = weights.sum() * 100

    def autopct(values):  # return the original percentage value, instead of the normalized one
        def my_autopct(pct):
            total = sum(values)
            return f'{pct * total:.2f}%'.rstrip('0').rstrip('.')

        return my_autopct

    plt.style.use('bmh')
    plt.pie(weights, labels=weights.index, autopct=autopct(weights), pctdistance=0.85)
    plt.title(f"Pesos da carteira - {total_pct:.2f}%")
    plt.tight_layout()
    plt.show()

    # backtest of the portfolio
    # get the returns of the selected securities
    returns = df[selected_securities].pct_change().dropna()
    portfolio_returns = returns @ weights.values
    acc_portfolio = (1 + portfolio_returns).cumprod()

    ibov = df[history.schema.security_manager.get("IND1!")]
    ibov = ibov.pct_change().dropna()

    cdi = df[history.schema.security_manager.get("IPCA2029")]
    cdi = cdi.pct_change().dropna()

    spy = df[history.schema.security_manager.get("ES=F")]
    spy = spy.pct_change().dropna()

    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')

    def custom_date_formatter(x, pos=None):
        return mdates.num2date(x).strftime('%b - %Y').capitalize()

    # Criar o formatter usando a função personalizada
    months_fmt = FuncFormatter(custom_date_formatter)

    # plot accumulated returns graph, and the portfolio accumulated returns, on the same graph
    plt.figure(figsize=(10, 6))
    plt.plot(acc_portfolio, color='gold', linestyle=':', linewidth=2, label='Portfolio')
    plt.plot((1 + ibov).cumprod(), color='blue', linestyle='-', linewidth=2, label='Ibovespa', alpha=0.6)
    plt.plot((1 + cdi).cumprod(), color='green', linestyle='-', linewidth=2, label='CDI', alpha=0.6)
    plt.plot((1 + spy).cumprod(), color='crimson', linestyle='-', linewidth=2, label='SPY', alpha=0.6)

    plt.title('Retornos Acumulados', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Retorno Acumulado', fontsize=14)
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.gca().xaxis.set_major_formatter(months_fmt)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().tick_params(axis='x', labelsize=9)

    plt.show()

    # Drawdown
    drawdown = 1 - acc_portfolio / acc_portfolio.cummax()

    # Maximum drawdown until t - Deepest lake
    max_drawdown = drawdown.cummax()

    # Plot drawdown (upside down, negative) + horizontal line for max drawdown
    plt.figure(figsize=(10, 6))
    plt.plot(-drawdown, color='red', linewidth=2, label='Drawdown')
    # plt.axhline(-max_drawdown, color='black', linestyle='--', linewidth=1, label='Max Drawdown')
    plt.plot(-max_drawdown, color='black', linestyle='--', linewidth=1, label='Max Drawdown')
    plt.title('Drawdown', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=14)
    plt.ylabel('Drawdown', fontsize=14)
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True, borderpad=1)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.gca().xaxis.set_major_formatter(months_fmt)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.gca().tick_params(axis='x', labelsize=9)

    plt.show()

    # Metrics to print:
    # Annual return, annualized volatility, sharpe, % months positive, max monthly reeturn, min monthly return
    annual_return = acc_portfolio[-1] - 1
    annual_volatility = portfolio_returns.std() * np.sqrt(len(portfolio_returns))
    annual_rf_rate = (1 + cdi).prod() - 1
    sharpe = (annual_return - annual_rf_rate) / annual_volatility
    months_positive = portfolio_returns.resample('M').mean().gt(0).mean()
    max_monthly_return = (1 + portfolio_returns).resample('M').prod().max() - 1
    min_monthly_return = (1 + portfolio_returns).resample('M').prod().min() - 1

    print(f"Retorno anual: {annual_return:.2%}\n" +
          f"Volatilidade anualizada: {annual_volatility:.2%}\n" +
          f"Sharpe: {sharpe:.2f}\n" +
          f"% meses positivos: {months_positive:.2%}\n" +
          f"Retorno mensal máximo: {max_monthly_return:.2%}\n" +
          f"Retorno mensal mínimo: {min_monthly_return:.2%}")

    # Save SPY, CDI, Ibovespa and Portfolio to single CSV
    to_save = pd.concat([(1 + spy).cumprod(), (1 + cdi).cumprod(), (1 + ibov).cumprod(), acc_portfolio], axis=1)
    to_save.columns = ['SPY', 'CDI', 'Ibovespa', 'Portfolio']
    to_save.to_csv('portfolio.csv')


if __name__ == "__main__":
    main()
