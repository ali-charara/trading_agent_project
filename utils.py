import pandas as pd
import yfinance as yf


def load_stocks(
    ticker: tuple[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:

    return yf.download(ticker, start_date, end_date)
