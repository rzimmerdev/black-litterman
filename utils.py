import numpy as np
import pandas as pd
import cvxpy as cp
from scipy.optimize import optimize
import matplotlib.pyplot as plt
from scipy.stats import norm


def dirichlet_sampling(alpha, columns, num_samples=20000):
    """
    Sample random portfolios from a Dirichlet distribution.

    Args:
        alpha (array-like): Concentration parameters of the Dirichlet distribution.
        columns (list): Names of the stocks in the portfolio.
        num_samples (int): Number of random portfolios to sample.

    Returns:
        DataFrame: Sampled portfolios where each row is a portfolio and each column is a stock.
    """
    # Generate samples from the Dirichlet distribution
    samples = np.random.dirichlet(alpha, num_samples)

    # Convert to a DataFrame for easier handling
    return pd.DataFrame(samples, columns=columns)


def dirichlet_weights(mu, sigma, alpha, num_samples=20000):
    """
    Estimate the mean and covariance of portfolios sampled from a Dirichlet distribution.
    Find the portfolio closest to the optimal (as given by mean-variance optimization).

    Args:
        mu (Series): Expected returns of assets.
        sigma (DataFrame): Covariance matrix of assets.
        alpha (array-like): Concentration parameters for the Dirichlet distribution.
        num_samples (int): Number of portfolios to sample.

    Returns:
        Series: Portfolio weights closest to the optimized portfolio.
    """
    # Sample portfolios from the Dirichlet distribution
    df = dirichlet_sampling(alpha, mu.index, num_samples)

    # Calculate mean and covariance of each sampled portfolio
    sigma_np = sigma.values
    means = df.dot(mu)
    # sigma is n x n
    # df is num_samples x n
    covariances = df.apply(lambda x: x.dot(sigma_np).dot(x), axis=1)

    # Find the portfolio closest to the optimal
    optimal_weights = mean_variance_optim(mu, sigma)
    optimal_mean = optimal_weights.dot(mu)
    optimal_covariance = optimal_weights.dot(sigma).dot(optimal_weights)

    # Calculate distance to optimal portfolio
    distances = np.abs(means - optimal_mean) + np.abs(covariances - optimal_covariance)

    # Find the portfolio with the minimum distance
    min_index = distances.idxmin()

    return df.loc[min_index]


def black_litterman(P, q, Omega, pi, sigma, tau=0.025):
    """
    Perform the Black-Litterman model to generate expected returns.
    Accepts two sets of inputs: the market-implied returns and the investor's views.

    Args:
        P (DataFrame): Pick matrix
        q (DataFrame): Views
        Omega (DataFrame): Views uncertainty
        pi (Series): Equilibrium excess returns
        sigma (DataFrame): Covariance matrix of assets
        tau (float): Scaling factor

    Returns:
        Series: Expected returns
    """
    # Step 1: Adjust the covariance matrix with the scaling factor tau
    tau_sigma = tau * sigma

    # Step 2: Calculate the posterior estimate of the mean
    # First, calculate the middle term for the posterior distribution
    middle_term = np.linalg.inv(np.dot(np.dot(P.T, np.linalg.inv(Omega)), P) + np.linalg.inv(tau_sigma))

    # Then, calculate the posterior expected returns
    term1 = np.dot(np.dot(middle_term, P.T), np.dot(np.linalg.inv(Omega), q))
    term2 = np.dot(middle_term, np.dot(np.linalg.inv(tau_sigma), pi))

    # Final estimate of the mean
    theta = term1 + term2

    return pd.Series(theta, index=pi.index)


def mean_variance_optim(mu, sigma, gamma=1, max_stocks=None, var_bound=None, conf=0.95):
    """
    Perform mean-variance optimization with a cardinality constraint.

    Args:
        mu (Series): Expected returns
        sigma (DataFrame): Covariance matrix of assets
        gamma (float): Risk aversion parameter
        max_stocks (int): Maximum number of stocks in the portfolio
        var_bound (float): Variance bound
        conf (float): Confidence level for the variance bound

    Returns:
        Series: Optimal weights
    """
    n = len(mu)
    mu = mu.values.reshape(-1)

    w = cp.Variable(n)
    y = cp.Variable(n, boolean=True)

    # Objective function: Mean-variance optimization
    objective = cp.Minimize(
        0.5 * cp.quad_form(w, sigma) - gamma * mu @ w)  # Minimize the negative of the utility function -> Maximize

    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= y,
    ]

    if isinstance(max_stocks, (int, float)):
        constraints.append(cp.sum(y) <= max_stocks)  # Limit the number of selected stocks
    elif isinstance(max_stocks, dict):
        for key, value in max_stocks.items():
            constraints.append(cp.sum(y[key]) <= value)

    if var_bound is not None:
        z_score = norm.ppf(conf)
        std = cp.sqrt(cp.quad_form(w, sigma))
        var = mu @ w - z_score * std
        constraints.append(var >= var_bound)  # Limit the Value-at-Risk of current portfolio

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.ECOS_BB)

    return w.value


def main():
    # plot 3 pies side by side
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    mu = pd.Series([0.12, 0.10, 0.15, 0.08], index=['AAPL', 'GOOGL', 'MSFT', 'AMZN'])
    sigma = pd.DataFrame([[0.10, 0.02, 0.03, 0.01],
                          [0.02, 0.08, 0.01, 0.02],
                          [0.03, 0.01, 0.09, 0.01],
                          [0.01, 0.02, 0.01, 0.07]])

    gamma = 1
    max_stocks = 2

    weights = mean_variance_optim(mu, sigma, gamma, max_stocks)

    # No BL weights
    print(weights)

    # Plot pie chart
    # Remove weights that are close to zero
    weights = pd.Series(weights, index=mu.index)
    weights = weights[weights > 1e-4]
    axs[0].pie(weights, labels=weights.index, autopct='%1.1f%%')

    # Using BL
    # P is the pick matrix, that is, the matrix that maps the views to the assets
    # The rows of P are the views, rows must sum to zero
    # for example, first row: first column will outperform at ratio of 1 to -1 to the last column  shape: n x m, where n is the number of views and m is the number of assets
    P = pd.DataFrame([[1, 0, 0, -1],
                      [0, 1, 0, -1]])

    # q is the views, that is, the expected returns of the assets given the views
    q = pd.Series([0.02, 0.03])

    # Omega is the uncertainty matrix of the views (diagonal matrix)
    # That is, the variance for each view row
    Omega = pd.DataFrame([[0.01, 0],
                          [0, 0.02]])

    # pi is the equilibrium excess returns, this is the expected returns without the views (the prior, or previous mu)
    pi = pd.Series([0.12, 0.10, 0.15, 0.08])

    # tau is the scaling factor, or the impact for the views on the expected returns
    tau = 1

    mu_bl = black_litterman(P, q, Omega, pi, sigma, tau)
    weights_bl = mean_variance_optim(mu_bl, sigma, gamma, max_stocks)

    # BL weights
    print(weights_bl)

    # Plot pie chart
    # Remove weights that are close to zero
    weights_bl = pd.Series(weights_bl, index=mu.index)
    weights_bl = weights_bl[weights_bl > 1e-4]
    axs[1].pie(weights_bl, labels=weights_bl.index, autopct='%1.1f%%')

    # Dirichlet weights
    alpha = [1, 1, 1, 1]
    weights_dirichlet = dirichlet_weights(mu, sigma, alpha)
    print(weights_dirichlet)

    # Plot pie chart
    # Remove weights that are close to zero
    weights_dirichlet = pd.Series(weights_dirichlet, index=mu.index)
    weights_dirichlet = weights_dirichlet[weights_dirichlet > 1e-4]
    axs[2].pie(weights_dirichlet, labels=weights_dirichlet.index, autopct='%1.1f%%')
    plt.show()


if __name__ == "__main__":
    main()
