from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from gym import Env, spaces

from constants import (
    DEFAULT_TICKER,
    INITIAL_FUND,
    MAX_SHARES_PER_STOCK,
    START_DATE,
    TRADING_WINDOW_DAY_DURATION,
    TRANSACTION_FEE_PERCENTAGE,
    TRANSITION_DATE,
    FIGSIZE,
)
from indicators import add_technical_indicators
from utils import load_stocks


class TrainStockEnvironment(Env):
    def __init__(
        self,
        ticker: tuple[str] = DEFAULT_TICKER,
        start_date: str = START_DATE,
        end_date: str = TRANSITION_DATE,
        initial_fund: float = INITIAL_FUND,
        max_shares_per_stock: int = MAX_SHARES_PER_STOCK,
        window_day_duration: int = TRADING_WINDOW_DAY_DURATION,
    ) -> None:
        # gym env
        super().__init__()
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
        self.current_time = max(2, window_day_duration)
        self.window_day_duration = window_day_duration
        self.max_shares_per_stock = max_shares_per_stock
        self.b = None
        self.h = None
        self.prev_b = None
        self.prev_h = None

        # monitoring
        self.portfolio_history = []
        self.shares_history = []
        self.balance_history = []
        self.reward_history = []

    def _update_history(self) -> None:
        h_copy = self.h.copy()
        for t in range(
            self.current_time,
            min(
                self.current_time + self.window_day_duration,
                self.stocks_df.shape[0] - 1,
            ),
        ):
            self.portfolio_history.append(
                self.get_portfolio_value(self.b, self._get_prices(t), self.h)
            )
            self.shares_history.append(h_copy)
            self.balance_history.append(self.b)

    def _sell_stock(
        self, stock_index: int, sell_action: int, stock_prices: np.ndarray
    ) -> None:
        """Performs sell action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the sell action
            sell_action (int): associated sell action
            stock_prices (np.ndarray): stock prices array
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
        self, stock_index: int, buy_action: int, stock_prices: np.ndarray
    ) -> None:
        """Performs buy action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the buy action
            buy_action (int): associated buy action
            stock_prices (np.ndarray): stock prices array
        """
        available_amount = self.b // (
            stock_prices[stock_index] * (1 + TRANSACTION_FEE_PERCENTAGE)
        )
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

    def _get_reward(self, prev_prices: np.ndarray, prices: np.ndarray) -> float:
        return self.get_portfolio_value(
            self.b, prices, self.h
        ) - self.get_portfolio_value(self.prev_b, prev_prices, self.prev_h)

    def _get_info(self) -> dict:
        return {}

    @staticmethod
    def get_portfolio_value(
        balance: float, prices: np.ndarray, shares: np.ndarray
    ) -> float:
        return balance + prices @ shares

    def reset(self) -> np.ndarray:
        self.current_time = max(2, self.window_day_duration)
        self.b = self.initial_fund
        self.h = np.zeros(len(self.ticker))

        self.portfolio_history = [self.initial_fund]
        self.shares_history = [self.h.copy()]
        self.balance_history = [self.b]
        self.reward_history = []

        return np.concatenate((self._get_state(), self._get_observation()))

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        prev_prices = self._get_prices(self.current_time)
        self.prev_b = self.b
        self.prev_h = self.h.copy()

        # perform action
        action = np.round(action * self.max_shares_per_stock)
        argsort_actions = np.argsort(action)
        sell_indexes = argsort_actions[: np.where(action < 0)[0].shape[0]]
        buy_indexes = argsort_actions[::-1][: np.where(action >= 0)[0].shape[0]]
        for index in sell_indexes:
            self._sell_stock(index, action[index], prev_prices)

        for index in buy_indexes:
            self._buy_stock(index, action[index], prev_prices)

        self._update_history()

        self.current_time += self.window_day_duration
        self.current_time = min(self.current_time, self.stocks_df.shape[0] - 1)

        reward = self._get_reward(prev_prices, self._get_prices(self.current_time))
        self.reward_history.append(reward)

        return (
            np.concatenate((self._get_state(), self._get_observation())),
            reward,
            self.current_time >= self.stocks_df.shape[0] - 1,
            self._get_info(),
        )

    def render(
        self, agent_name: str, benchmark_weights: Optional[dict[str, np.ndarray]] = None
    ) -> None:
        fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)

        fig.suptitle(
            f"Investment Strategy of {agent_name} agent rebalancing every {self.window_day_duration} day(s)"
        )

        axes[0].plot(
            (np.array(self.portfolio_history) - self.initial_fund) / self.initial_fund,
            label=agent_name,
        )
        axes[0].set_title("Portfolio return")
        axes[0].set_xlabel("day")
        axes[0].set_ylabel("Cumulative Return")

        if benchmark_weights is not None:
            for method, weights in benchmark_weights.items():
                axes[0].plot(
                    (
                        self.stocks_df["Adj Close"]
                        / self.stocks_df["Adj Close"].iloc[0]
                        - 1
                    )
                    @ weights,
                    label=method,
                )

            axes[0].legend(loc=2)

        if hasattr(self, "turbulence_df"):
            ax_turbulence = axes[0].twinx()
            ax_turbulence.plot(self.turbulence_df, "r--", alpha=0.25)
            ax_turbulence.set_ylabel("turbulence")

        axes[1].plot(np.vstack(self.shares_history))
        axes[1].set_title("Shares hold")
        axes[1].set_xlabel("day")
        axes[1].set_ylabel("Shares")
        axes[1].legend(self.ticker, loc=2)

        ax_balance = axes[1].twinx()
        ax_balance.plot(self.balance_history, "r--", alpha=0.25)
        ax_balance.set_ylabel("Remaining balance")

        plt.show()
