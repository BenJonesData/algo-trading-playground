import yfinance as yf
import pandas as pd
from helper_functions.indicators import rsi
from loguru import logger
from typing import List, Callable, Union
from functools import wraps


def get_price_data_and_rsi(
    tickers:  Union[str, List[str]],
    start_date: str,
    end_date: str,
    interval: str,
    rsi_periods:  Union[int, List[int]],
    remove_zeros: bool = True,
    logger_batch_size: int = None,
) -> pd.DataFrame:
    """
    Downloads price data for a given time period and interval, and calculates 
    a return column and RSI (Relative Strength Index) columns for each
    specified period.

    Args:
        - tickers (str or List[str]): The ticker symbol(s) to obtain data for.
        - start_date (str): The start date of the time period (format
        'YYYY-MM-DD').
        - end_date (str): The end date of the time period (format
        'YYYY-MM-DD').
        - interval (str): The granularity of the data (e.g., '1d', '1h', etc.).
        - rsi_periods (int or List[int]): The period(s) to calculate the RSI
        for. If multiple periods are provided, multiple RSI columns will be
        added.
        - remove_zeros (bool, optional, default=True): Whether to remove rows
        where the RSIs could not be calculated due to insufficient data.
        - logger_batch_size (int, optional, default=None): Number of tickers
        to log in each batch for progress tracking. If None, the built in
        logging from yfinance is used on each individual download.

    Raises:
        ValueError: If `logger_batch_size` exceeds the number of tickers
        provided.

    Returns:
        pd.DataFrame: A DataFrame indexed by ticker and date, containing price
        data columns and additional RSI columns. Each RSI column will be
        labeled as 'RSI_<period>', where <period> is the RSI period used for
        calculation.

    Notes:
        - If `remove_zeros=True`, the function will remove the first 
        `max(rsi_periods)` rows for each ticker, where there is insufficient
        data to calculate the RSI.
        - If `remove_zeros=False`, the initial rows of each RSI column will
        contain zeros until there is enough historical data to calculate the
        RSI for the corresponding period. The number of zeros will match the
        length of the RSI period. This will also mean that the first entry in
        the Return column for each ticker will be `NaN`.
    """
    
    if isinstance(tickers, str):
        tickers = [tickers]

    if isinstance(rsi_periods, int):
        rsi_periods = [rsi_periods]

    if logger_batch_size is not None:
        if logger_batch_size > len(tickers):
            raise ValueError(
                "logger_batch_size cannot exceed the number of tickers."
            )
        yfinance_progress = False
        tickers_downloaded = 0
    else:
        yfinance_progress = True
    
    output_list =[]

    for t in tickers:
        data = yf.download(
            tickers=t, start=start_date, end=end_date, interval=interval, progress=yfinance_progress
        )

        data['Ticker'] = t
        for period in rsi_periods:
            data[f'rsi_{period}'] = rsi(data['Close'], period)

        if remove_zeros:
            data = data.iloc[max(rsi_periods):]
        
        if not yfinance_progress:
            tickers_downloaded += 1
            if  tickers_downloaded % logger_batch_size == 0:
                logger.info(f"{tickers_downloaded} of {len(tickers)} downloaded")
        
        output_list.append(data)
    
    output_df = pd.concat(output_list).reset_index()
    output_df = output_df.set_index(['Ticker', 'Date'])


    return(output_df)
