import matplotlib.pyplot as plt
import numpy as np
from constants import (
    DEFAULT_TICKER,
    END_DATE,
    START_DATE,
    TRANSACTION_COST_PERCENTAGE,
    WINDOW_DAY_DURATION,
)
from indicators import add_technical_indicators
from utils import load_stocks


class TradingStockEnvironment:
    def __init__(
        self,
        ticker: tuple[str] = DEFAULT_TICKER,
        start_date: str = START_DATE,
        end_date: str = END_DATE,
        window_day_duration: int = WINDOW_DAY_DURATION,
    ) -> None:
        self.ticker = ticker
        self.stocks_df = add_technical_indicators(
            load_stocks(ticker=ticker, start_date=start_date, end_date=end_date)
        ).reset_index()

        # investment strategy
        self.current_time = window_day_duration
        self.window_day_duration = window_day_duration
        self.b = None
        self.h = None
        self.prev_portfolio_value = None

        # monitoring
        self.portfolio_history = []

    def _update_history(self) -> None:
        self.portfolio_history.extend(
            [
                self.get_portfolio_value(t)
                for t in range(
                    self.current_time, self.current_time + self.window_day_duration
                )
            ]
        )

    def _get_state(self) -> np.ndarray:
        return np.concatenate(
            ([self.b], self.stocks_df["Adj Close"].loc[self.current_time], self.h)
        )

    def _get_observation(self) -> np.ndarray:
        return (
            self.stocks_df[["macd", "rsi", "cci", "dx"]]
            .loc[self.current_time]
            .to_numpy()
        )

    def _get_reward(self, action, prices) -> float:
        return (
            self.get_portfolio_value(self.current_time)
            - self.prev_portfolio_value
            - action @ prices * TRANSACTION_COST_PERCENTAGE
        )

    def get_portfolio_value(self, t) -> float:
        return self.b + self.stocks_df["Adj Close"].loc[t] @ self.h

    def reset(self, initial_fund: int):
        self.current_time = self.window_day_duration
        self.b = initial_fund
        self.h = np.zeros(len(self.ticker))
        self.prev_portfolio_value = self.get_portfolio_value(self.current_time)

        self._update_history()

        return self._get_state(), self._get_observation()

    def step(self, action):
        prev_prices = self.stocks_df["Adj Close"].loc[self.current_time].to_numpy()

        self.h += action
        buy_indexes = np.where(action > 0)
        sell_indexes = np.where(action < 0)
        self.b += (
            action[sell_indexes] @ prev_prices[sell_indexes]
            - action[buy_indexes] @ prev_prices[buy_indexes]
        )
        assert self.b >= 0

        self.current_time += self.window_day_duration
        reward = self._get_reward(action, prev_prices)
        self.prev_portfolio_value = self.get_portfolio_value(self.current_time)

        self._update_history()

        return (
            self._get_state(),
            self._get_observation(),
            reward,
        )

    def render(self) -> None:
        plt.plot(self.portfolio_history)

        plt.show()
