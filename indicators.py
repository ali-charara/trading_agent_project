import pandas as pd
from stockstats import StockDataFrame


def add_technical_indicators(market_dataframe: pd.DataFrame):
    stocks_dataframe = market_dataframe.swaplevel(i=0, j=1, axis=1)
    ticker_list = stocks_dataframe.columns.levels[0].tolist()

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    for ticker in ticker_list:
        ## macd
        stock = StockDataFrame.retype(stocks_dataframe[ticker])
        temp_macd = stock["macd"]
        temp_macd = pd.DataFrame(temp_macd)
        macd = pd.concat((macd, temp_macd), ignore_index=True, axis=1)
        ## rsi
        temp_rsi = stock["rsi_30"]
        temp_rsi = pd.DataFrame(temp_rsi)
        rsi = pd.concat((rsi, temp_rsi), ignore_index=True, axis=1)
        ## cci
        temp_cci = stock["cci_30"]
        temp_cci = pd.DataFrame(temp_cci)
        cci = pd.concat((cci, temp_cci), ignore_index=True, axis=1)
        ## adx
        temp_dx = stock["dx_30"]
        temp_dx = pd.DataFrame(temp_dx)
        dx = pd.concat((dx, temp_dx), ignore_index=True, axis=1)

    market_dataframe = market_dataframe.join(
        pd.DataFrame(
            macd.to_numpy(),
            columns=pd.MultiIndex.from_product([["macd"], ticker_list]),
            index=market_dataframe.index,
        )
    )
    market_dataframe = market_dataframe.join(
        pd.DataFrame(
            rsi.to_numpy(),
            columns=pd.MultiIndex.from_product([["rsi"], ticker_list]),
            index=market_dataframe.index,
        )
    )
    market_dataframe = market_dataframe.join(
        pd.DataFrame(
            cci.to_numpy(),
            columns=pd.MultiIndex.from_product([["cci"], ticker_list]),
            index=market_dataframe.index,
        )
    )
    market_dataframe = market_dataframe.join(
        pd.DataFrame(
            dx.to_numpy(),
            columns=pd.MultiIndex.from_product([["dx"], ticker_list]),
            index=market_dataframe.index,
        )
    )

    return market_dataframe
