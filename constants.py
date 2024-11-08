# fmt: off
DEFAULT_TICKER = ("^DJI", "^IXIC", "AMZN", "GOOGL",
                  "AAPL", "PFE", "AZN", "JNJ",
                  "SNY")

# fmt: on
#! data string format must be "YYYY-MM-DD"
START_DATE = "2009-01-01"
TRANSITION_DATE = "2016-01-01"
END_DATE = "2020-08-05"

# investment strategy
INITIAL_FUND = 1_000_000
MAX_SHARES_PER_STOCK = 100
TRANSACTION_FEE_PERCENTAGE = 0.01
TURBULENCE_THRESHOLD = 120
TRADING_WINDOW_DAY_DURATION = 1

# plot
FIGSIZE = (20, 5)
