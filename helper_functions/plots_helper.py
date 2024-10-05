import plotly.graph_objects as go
import pandas as pd


def create_candlestick(data: pd.DataFrame) -> go.Candlestick:
    """
    Create a candlestick chart object from a financial DataFrame.

    This function constructs a Plotly Candlestick object using financial data
    from the input DataFrame. The DataFrame can have any time granularity
    (e.g., daily, hourly), but it must be indexed by `DatetimeIndex`.

    Args:
        data (pd.DataFrame): The input DataFrame. The DataFrame must meet the following requirements:
            - It must be indexed by a `DatetimeIndex` (i.e., the index must be of type `datetime`).
            - It must contain the following columns:
                - 'Open' (float): The opening price for each time period.
                - 'High' (float): The highest price for each time period.
                - 'Low' (float): The lowest price for each time period.
                - 'Close' (float): The closing price for each time period.

    Returns:
        go.Candlestick: A Plotly Candlestick object representing the financial data.

    Raises:
        ValueError: If the DataFrame is not indexed by a `DatetimeIndex` or if any
        of the required columns ('Open', 'High', 'Low', 'Close') are missing.

    Example:
        >>> import pandas as pd
        >>> import plotly.graph_objects as go
        >>> df = pd.DataFrame({
        ...     'Open': [100.0, 102.5, 105.0],
        ...     'High': [101.0, 103.0, 106.0],
        ...     'Low': [99.0, 101.5, 104.5],
        ...     'Close': [100.5, 102.0, 105.5]
        ... }, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
        >>> candlestick = create_candlestick(df)
    """
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("The DataFrame must be indexed by a `DatetimeIndex`.")

    required_columns = ["Open", "High", "Low", "Close"]

    missing_columns = [
        col for col in required_columns if col not in data.columns
    ]  # find required columns not present in data
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    return go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
    )
