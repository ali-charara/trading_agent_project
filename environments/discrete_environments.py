import numpy as np
from gym import spaces

from constants import (
    DEFAULT_TICKER,
    END_DATE,
    INITIAL_FUND,
    START_DATE,
    TRADING_WINDOW_DAY_DURATION,
    TRANSITION_DATE,
    TURBULENCE_THRESHOLD,
)
from utils import number_to_base

from .trade_environment import TradingStockEnvironment
from .train_environment import TrainStockEnvironment

MAX_SHARES_PER_STOCK = 2


class DiscreteStockEnvironment(TrainStockEnvironment):
    """This class derived from TrainStockEnvironment allows to simulate an
    environment aiming at building a portfolio over the stocks of interest
    over a time window.

    The action space of this specific environment is discrete as in the
    Reinforcement Learning Course of CentraleSupélec. However, we'll deal
    with continuous State Representation instead of tiling encoding due to
    the high dimension of the state space.
    """

    def __init__(
        self,
        ticker: str = DEFAULT_TICKER,
        start_date: str = START_DATE,
        end_date: str = TRANSITION_DATE,
        initial_fund: float = INITIAL_FUND,
        max_shares_per_stock: int = MAX_SHARES_PER_STOCK,
        window_day_duration: int = TRADING_WINDOW_DAY_DURATION,
    ) -> None:
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_fund=initial_fund,
            max_shares_per_stock=max_shares_per_stock,
            window_day_duration=window_day_duration,
        )

        self.action_space = spaces.Discrete(
            (2 * max_shares_per_stock + 1) ** len(ticker)
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        final_action = (
            np.array(
                number_to_base(
                    action, 2 * self.max_shares_per_stock + 1, len(self.ticker)
                )
            )
            - self.max_shares_per_stock
        ) / self.max_shares_per_stock

        return super().step(final_action)


class DiscreteTradingEnvironment(TradingStockEnvironment):
    def __init__(
        self,
        ticker: str = DEFAULT_TICKER,
        start_date: str = TRANSITION_DATE,
        end_date: str = END_DATE,
        initial_fund: float = INITIAL_FUND,
        max_shares_per_stock: int = MAX_SHARES_PER_STOCK,
        turbulence_threshold: float = TURBULENCE_THRESHOLD,
        window_day_duration: int = TRADING_WINDOW_DAY_DURATION,
    ) -> None:
        super().__init__(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_fund=initial_fund,
            max_shares_per_stock=max_shares_per_stock,
            turbulence_threshold=turbulence_threshold,
            window_day_duration=window_day_duration,
        )

        self.action_space = spaces.Discrete(
            (2 * max_shares_per_stock + 1) ** len(ticker)
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        final_action = (
            np.array(
                number_to_base(
                    action, 2 * self.max_shares_per_stock + 1, len(self.ticker)
                )
            )
            - self.max_shares_per_stock
        ) / self.max_shares_per_stock

        return super().step(final_action)
