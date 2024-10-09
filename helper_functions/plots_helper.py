import plotly.graph_objects as go
import pandas as pd
import matplotlib as plt


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
        raise ValueError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    return go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
    )


def plot_position(
    price_data: pd.DataFrame, position_data: pd.DataFrame, asset_label: str
) -> None:
    """
    Plots asset price and position size over time using a dual-axis plot, where the primary axis shows the asset price and the secondary axis shows the position size.

    Args:
        price_data (pd.DataFrame): A time series DataFrame containing the asset price, indexed by a `DatetimeIndex`.
        position_data (pd.DataFrame): A time series DataFrame containing the position size, indexed by a `DatetimeIndex`.
        asset_label (str): A string representing the asset's name, which is used in the plot's title.

    Raises:
        ValueError: If either `price_data` or `position_data` is not indexed by a `DatetimeIndex`.

    Notes:
        - The function creates a plot where the asset price is plotted on the left y-axis in red, and the position size on the right y-axis in blue.
        - Ensure that both input DataFrames are aligned on the same time axis for accurate plotting.

    Example:
        >>> plot_position(price_data=df_price, position_data=df_position, asset_label="AAPL")
    """

    if not (
        isinstance(price_data.index, pd.DatetimeIndex)
        & isinstance(position_data.index, pd.DatetimeIndex)
    ):
        raise ValueError(
            "Both `price_data` and `position_data` must be indexed by a `DatetimeIndex`."
        )

    fig, ax1 = plt.subplots()
    ax1.plot(price_data, color="red")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Value (USD)", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    ax2 = ax1.twinx()
    ax2.plot(position_data, color="blue")
    ax2.set_ylabel("Position Size", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    plt.title(f"{asset_label} Price and Strategy Position Size Over Time")
    plt.show()
