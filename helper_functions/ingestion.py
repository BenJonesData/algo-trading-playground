import yfinance as yf
import pandas as pd
from helper_functions.indicators import rsi
from loguru import logger


def get_price_data_and_rsi(
    tickers: list,
    period: int,
    start_date: str,
    end_date: str,
    interval: str,
    remove_zeros: bool = True,
    logger_batch_size: int = None,
) -> pd.DataFrame:
    

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
        data['Return'] = data['Close'].pct_change()
        data['rsi'] = rsi(data['Close'], period)

        if remove_zeros:
            data = data.iloc[period:]
        
        if not yfinance_progress:
            tickers_downloaded += 1
            if  tickers_downloaded % logger_batch_size == 0:
                logger.info(f"{tickers_downloaded} of {len(tickers)} downloaded")
        
        output_list.append(data)
    
    output_df = pd.concat(output_list).reset_index()
    output_df = output_df.set_index(['Ticker', 'Date'])


    return(output_df)