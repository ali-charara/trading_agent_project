import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize


def load_stocks(
    ticker: tuple[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    stocks_dataframe = yf.download(ticker, start_date, end_date)
    if stocks_dataframe.columns.nlevels == 1:
        stocks_dataframe.columns = pd.MultiIndex.from_product(
            [stocks_dataframe.columns, ticker]
        )

    return stocks_dataframe


def number_to_base(n, b, l):
    if n == 0:
        return [0] * l

    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits) < l:
        digits.append(0)

    return digits[::-1]


def default_constraint(w, linear_constraint, bounds):
    n = w.shape[0]
    if linear_constraint is None:
        linear_constraint = optimize.LinearConstraint([[1] * n], [1], [1])
    if bounds is None:
        bounds = optimize.Bounds([0] * n, [1] * n)

    return linear_constraint, bounds


def compute_mv_weights(
    w: np.ndarray, cov_matrix: np.ndarray, linear_constraint=None, bounds=None
) -> np.ndarray:
    linear_constraint, bounds = default_constraint(w, linear_constraint, bounds)

    def cost(w):
        return w.T.dot(cov_matrix).dot(w)

    res = optimize.minimize(
        fun=cost, x0=w, constraints=linear_constraint, bounds=bounds
    )
    w = res.x

    return w
