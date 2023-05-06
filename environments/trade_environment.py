import numpy as np

from constants import (
    DEFAULT_TICKER,
    END_DATE,
    INITIAL_FUND,
    MAX_SHARES_PER_STOCK,
    TRADING_WINDOW_DAY_DURATION,
    TRANSITION_DATE,
    TURBULENCE_THRESHOLD,
)
from indicators import calcualte_turbulence

from .train_environment import TrainStockEnvironment


class TradingStockEnvironment(TrainStockEnvironment):
    def __init__(
        self,
        ticker: tuple[str] = DEFAULT_TICKER,
        start_date: str = TRANSITION_DATE,
        end_date: str = END_DATE,
        initial_fund: float = INITIAL_FUND,
        max_shares_per_stock: int = MAX_SHARES_PER_STOCK,
        turbulence_threshold: float = TURBULENCE_THRESHOLD,
        window_day_duration: int = TRADING_WINDOW_DAY_DURATION,
    ) -> None:
        # gym env
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_fund=initial_fund,
            max_shares_per_stock=max_shares_per_stock,
            window_day_duration=window_day_duration,
        )

        # stock market
        self.turbulence_df = calcualte_turbulence(self.stocks_df)
        self.turbulence_threshold = turbulence_threshold

        # investment strategy
        self.turbulence = 0

    def _sell_stock(
        self, stock_index: int, sell_action: int, stock_prices: np.ndarray
    ) -> None:
        """Performs sell action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the sell action
            stock_prices (np.ndarray): stock prices array
            sell_action (int): associated sell action
        """
        if self.turbulence >= self.turbulence_threshold:
            sell_action = -self.h[stock_index]

        super()._sell_stock(stock_index, sell_action, stock_prices)

    def _buy_stock(
        self, stock_index: int, buy_action: int, stock_prices: np.ndarray
    ) -> None:
        """Performs buy action based on the sign of the action

        Args:
            stock_index (int): stock index concerned by the buy action
            stock_prices (np.ndarray): stock prices array
            buy_action (int): associated buy action
        """
        if self.turbulence < self.turbulence_threshold:
            super()._buy_stock(stock_index, buy_action, stock_prices)

    def reset(self) -> np.ndarray:
        next_state = super().reset()
        self.turbulence = self.turbulence_df.iloc[self.current_time]

        return next_state

    def step(self, action) -> tuple[np.ndarray, float, bool, dict]:
        if self.turbulence >= self.turbulence_threshold:
            action = -np.ones(len(self.ticker))

        next_state, reward, done, info = super().step(action)
        self.turbulence = self.turbulence_df.iloc[self.current_time]

        return next_state, reward, done, info
