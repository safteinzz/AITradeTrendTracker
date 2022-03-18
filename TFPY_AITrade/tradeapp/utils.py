# https://www.youtube.com/watch?v=jrT6NiM46jk matplotlib (no necesario en este caso)
from yahoo_fin import stock_info as si
from sklearn import preprocessing
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# ~Data Getter YAHOO
# =============================================================================
def get_dataYahoo(ticker, scaled = True, dropTicker = False, news = True, shuffle = True):
    # <!------------------- Extraccion ---------------------->
    # Comprobación tipos
    if isinstance(ticker, str):
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
        pass

    # Escalar valores 0-1 (mejorar rendimiento)
    if scaled:
        column_scaler = {}
        for columnName in df.columns:
            if columnName != 'ticker' and columnName != 'date':
                print(columnName)
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





