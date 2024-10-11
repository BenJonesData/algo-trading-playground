import pandas as pd
import numpy as np


def sma(values: np.array, n: int) -> pd.Series:
    """
    Computes the Simple Moving Average (SMA) of a NumPy array over a window of
    size `n`.

    Args:
        - values (np.array): A 1D array of numerical values.
        - n (int): The window size for the moving average.

    Returns:
        pd.Series: A Pandas Series containing the SMA, with NaN for the first
        `n-1` values.
    """
    return pd.Series(values).rolling(n).mean()


def _calculate_point_rsi(mean_upward: float, mean_downward: float) -> float:
    """
    Calculates the Relative Strength Index (RSI) based on mean upward and mean
    downward price movements.

    Args:
        - mean_upward (float): The average of upward price movements.
        - mean_downward (float): The average of downward price movements.

    Returns:
        float: The RSI value, ranging from 0 to 100. If `mean_downward` is 0,
        the function returns 100.
    """
    if mean_downward != 0:
        relative_strength = mean_upward / mean_downward
        return 100 - 100 / (1 + relative_strength)
    else:
        return 100


def rsi(prices: np.array, period: int) -> np.ndarray:
    """
    Calculates the Relative Strength Index (RSI) for a given price series.

    Args:
        - prices (pd.Series): The closing prices of the asset as a pandas
        Series.
        - period (int): The lookback period to calculate the RSI.

    Returns:
        np.ndarray: An array of RSI values, where the first `period` values
        will be zeros because RSI cannot be calculated until `period` data
        points are available.

    Raises:
        ValueError: If the input prices series contains fewer than `period`
        data points.

    Example:
        prices = pd.Series([44, 46, 45, 47, 44, 43, 42, 43, 44, 45])
        rsi_values = calculate_rsi(prices, 5)
    """
    prices = np.asarray(prices)
    deltas = np.diff(prices)

    upward_changes = np.where(deltas > 0, deltas, 0)
    downward_changes = np.where(deltas < 0, -deltas, 0)

    rsi = np.zeros(len(prices))

    mean_upward = np.mean(upward_changes[:period])
    mean_downward = np.mean(downward_changes[:period])

    rsi[period] = _calculate_point_rsi(mean_upward, mean_downward)

    for i in range(period + 1, len(prices)):
        mean_upward = (
            mean_upward * (period - 1) + upward_changes[i - 1]
        ) / period
        mean_downward = (
            mean_downward * (period - 1) + downward_changes[i - 1]
        ) / period

        rsi[i] = _calculate_point_rsi(mean_upward, mean_downward)

    return rsi
