# https://www.youtube.com/watch?v=jrT6NiM46jk matplotlib (no necesario en este caso)
from yahoo_fin import stock_info as si
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import date

def get_dataYahoo(ticker, scaled = True, dropTicker = False, news = True, shuffle = True, period = 0, interval = None):
    """_summary_

    Args:
        ticker (string_or_pd.DataFrame): Selected ticker for data extraction
        scaled (bool, optional): Scale values 0 to 1 for better performance. Defaults to True.
        dropTicker (bool, optional): Remove ticker column from dataframe. Defaults to False.
        news (bool, optional): Adquire news for training. Defaults to True.
        shuffle (bool, optional): Shuffle the data or not. Defaults to True.
        period (int, optional): Period of data search (0 week, 1 month, 2 year, 3 all). Defaults to 1.
        interval (_type_, optional): Interval of the data (weekly, monthly). Defaults to None.

    Raises:
        TypeError: _description_

    Returns:
        _type_: _description_
    """
    # <!------------------- Extraction ---------------------->
    # http://theautomatic.net/yahoo_fin-documentation/#methods
    #           
    # Type check
    if isinstance(ticker, str):
        # Get date range 
        endDateRange = date.today()
        if period != 3:
            if period == 0:
                startDateRange = endDateRange - pd.DateOffset(days=8)
            if period == 1:
                startDateRange = endDateRange - pd.DateOffset(months=1)
            if period == 2:
                startDateRange = endDateRange - pd.DateOffset(years=1)
            df = si.get_data(ticker, start_date = startDateRange , end_date = endDateRange)
        else:
            df = si.get_data(ticker, interval)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("Tipo no es 'str' o 'pd.DataFrame'")
    
    # <!------------------- Manipulation ---------------------->
    # Create column date instead of using it as index
    if "date" not in df.columns:
        df["date"] = df.index
        # Reset index
        df = df.reset_index(drop=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Delete duplicated rows (yahoo fin sometimes gives data of he same day twice)
    df = df.drop_duplicates(subset=["date"], keep=False)
    
    # Drop Ticker
    if dropTicker:
        del df["ticker"]

    # Scale values 0-1 (better perfomance)
    if scaled:
        column_scaler = {}
        for columnName in df.columns:
            if columnName != 'ticker' and columnName != 'date':
                scaler = preprocessing.MinMaxScaler()
                df[columnName] = scaler.fit_transform(np.expand_dims(df[columnName].values, axis=1))
                column_scaler[columnName] = scaler
            else:
                pass

    # Add news (deberia de sacar positividad de las noticias medias en referencia al tema para agregarlo como positividad de las noticias como linea adicional)
    if news:
        pass

    # Shuffle data
    if shuffle:
        pass

    return df