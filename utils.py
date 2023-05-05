import pandas as pd
import yfinance as yf


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
        return [0]

    digits = []
    while n:
        digits.append(int(n % b))
        n //= b

    while len(digits) < l:
        digits.append(0)

    return digits[::-1]
