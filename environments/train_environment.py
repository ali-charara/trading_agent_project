import matplotlib.pyplot as plt
import numpy as np
from constants import (DEFAULT_TICKER, INITIAL_FUND, MAX_SHARES_PER_STOCK,
                       START_DATE, TRADING_WINDOW_DAY_DURATION,
                       TRANSACTION_FEE_PERCENTAGE, TRANSITION_DATE)
from gym import Env, spaces
from indicators import add_technical_indicators
from utils import load_stocks


class TrainStockEnvironment(Env):
    def __init__(
        self,
        ticker: tuple[str] = DEFAULT_TICKER,
        start_date: str = START_DATE,
        end_date: str = TRANSITION_DATE,
        initial_fund: float = INITIAL_FUND,
        window_day_duration: int = TRADING_WINDOW_DAY_DURATION,
    ) -> None:
        # gym env
        super(TrainStockEnvironment).__init__()
        self.observation_space = spaces.Box(
            low=-3000, high=np.inf, shape=(len(ticker) * 6 + 1,)
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(ticker),))

        # stock market
        self.ticker = ticker
        self.stocks_df = add_technical_indicators(
            load_stocks(ticker=ticker, start_date=start_date, end_date=end_date)
        ).reset_index()

        # investment strategy
        self.initial_fund = initial_fund
        self.current_time = window_day_duration
        self.window_day_duration = window_day_duration
        self.b = None
        self.h = None
        self.prev_b = None
        self.prev_h = None

        # monitoring
        self.portfolio_history = []

    def _update_history(self) -> None:
        self.portfolio_history.extend(
            [
                self.get_portfolio_value(self.b, self._get_prices(t), self.h)
                for t in range(
                    self.current_time,
                    min(
                        self.current_time + self.window_day_duration,
                        self.stocks_df.shape[0] - 1,
                    ),
                )
            ]
        )

    def _sell_stock(
        self, stock_index: int, stock_prices: np.ndarray, sell_action: int
    ) -> None:
        """Performs sell action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the sell action
            stock_prices (np.ndarray): stock prices array
            sell_action (int): associated sell action
        """
        if self.h[stock_index] > 0:
            sold_shares = min(-sell_action, self.h[stock_index])
            self.b += (
                sold_shares
                * stock_prices[stock_index]
                * (1 - TRANSACTION_FEE_PERCENTAGE)
            )
            self.h[stock_index] -= sold_shares

    def _buy_stock(
        self, stock_index: int, stock_prices: np.ndarray, buy_action: int
    ) -> None:
        """Performs buy action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the buy action
            stock_prices (np.ndarray): stock prices array
            buy_action (int): associated buy action
        """
        available_amount = self.b // stock_prices[stock_index]
        bought_shares = min(available_amount, buy_action)

        self.b -= (
            bought_shares * stock_prices[stock_index] * (1 + TRANSACTION_FEE_PERCENTAGE)
        )
        self.h[stock_index] += bought_shares

    def _get_prices(self, timestamp: int) -> np.ndarray:
        return self.stocks_df["Adj Close"].loc[timestamp].to_numpy()

    def _get_state(self) -> np.ndarray:
        return np.concatenate(([self.b], self._get_prices(self.current_time), self.h))

    def _get_observation(self) -> np.ndarray:
        return (
            self.stocks_df[["macd", "rsi", "cci", "dx"]]
            .loc[self.current_time]
            .to_numpy()
        )

    def _get_reward(self, prev_prices, prices) -> float:
        return self.get_portfolio_value(
            self.b, prices, self.h
        ) - self.get_portfolio_value(self.prev_b, prev_prices, self.prev_h)

    def _get_info(self) -> dict:
        return {}

    @staticmethod
    def get_portfolio_value(balance, prices, shares) -> float:
        return balance + prices @ shares

    def reset(self) -> np.ndarray:
        self.current_time = self.window_day_duration
        self.b = self.initial_fund
        self.h = np.zeros(len(self.ticker))
        self.portfolio_history.append(self.initial_fund)

        return np.concatenate((self._get_state(), self._get_observation()))

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        prev_prices = self._get_prices(self.current_time)
        self.prev_b = self.b
        self.prev_h = self.h.copy()

        # perform action
        action = np.floor(action * MAX_SHARES_PER_STOCK)
        argsort_actions = np.argsort(action)
        sell_indexes = argsort_actions[: np.where(action < 0)[0].shape[0]]
        buy_indexes = argsort_actions[::-1][: np.where(action > 0)[0].shape[0]]
        for index in sell_indexes:
            self._sell_stock(index, prev_prices, action[index])

        for index in buy_indexes:
            self._buy_stock(index, prev_prices, action[index])

        self._update_history()

        self.current_time += self.window_day_duration
        self.current_time = min(self.current_time, self.stocks_df.shape[0] - 1)

        return (
            np.concatenate((self._get_state(), self._get_observation())),
            self._get_reward(prev_prices, self._get_prices(self.current_time)),
            self.current_time >= self.stocks_df.shape[0] - 1,
            self._get_info(),
        )

    def render(self) -> None:
        plt.plot(self.portfolio_history)
        
        plt.show()
