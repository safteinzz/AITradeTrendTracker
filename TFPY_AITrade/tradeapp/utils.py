# https://www.youtube.com/watch?v=jrT6NiM46jk matplotlib (no necesario en este caso)
from yahoo_fin import stock_info as si
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import date

def get_dataYahoo(ticker, scaled = True, dropTicker = False, news = True, shuffle = True, period = 0):
    """Funcion para extraer los datos del ticker de la libreria de yahoo_fin

    Args:
        ticker (string_or_pd.DataFrame): Selected ticker for data extraction
        scaled (bool, optional): Scale values 0 to 1 for better performance. Defaults to True.
        dropTicker (bool, optional): Remove ticker column from dataframe. Defaults to False.
        news (bool, optional): Adquire news for training. Defaults to True.
        shuffle (bool, optional): Shuffle the data or not. Defaults to True.
        period (int, optional): Period of data search (0 week, 1 month, 2 year, 3 all). Defaults to 1.

    Raises:
        TypeError: Throws error if is the type is wrong.

    Returns:
        pd.DataFrame: Data of the seleccted ticker.
    """
    # <!------------------- Extraccion ---------------------->
    # Get date range
    endDateRange = date.today()
    if period != 3:
        if period == 0:
            startDateRange = endDateRange - pd.DateOffset(days=7)
        if period == 1:
            startDateRange = endDateRange - pd.DateOffset(months=1)
        if period == 2:
            startDateRange = endDateRange - pd.DateOffset(years=1)
        
    # Comprobación tipos
    if isinstance(ticker, str):
        if period != 3:
            df = si.get_data(ticker, start_date = startDateRange , end_date = endDateRange)
        else:
            df = si.get_data(ticker)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("Tipo no es 'str' o 'pd.DataFrame'")
    
    # <!------------------- Manipulacion ---------------------->
    # Crear columna data en lugar de usarlo como index
    if "date" not in df.columns:
        df["date"] = df.index

    # Drop Ticker
    if dropTicker:
        del df["ticker"]

    # Escalar valores 0-1 (mejorar rendimiento)
    if scaled:
        column_scaler = {}
        for columnName in df.columns:
            if columnName != 'ticker' and columnName != 'date':
                scaler = preprocessing.MinMaxScaler()
                df[columnName] = scaler.fit_transform(np.expand_dims(df[columnName].values, axis=1))
                column_scaler[columnName] = scaler
            else:
                pass

    # Hay que agregar datos como noticias, deberia de sacar positividad de las noticias medias en referencia al tema para agregarlo como positividad de las noticias como linea adicional
    if news:
        pass

    # Mezclar datos para mejor confidence
    if shuffle:
        pass

    return df





