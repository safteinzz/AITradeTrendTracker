# https://www.youtube.com/watch?v=jrT6NiM46jk matplotlib (not needed but maybe usefull in the future)
from regex import P
from yahoo_fin import stock_info as si
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import date, datetime
from GoogleNews import GoogleNews
from urllib.parse import urlparse
import re
from talib import BBANDS


import os
from math import sqrt


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error 

from .models import New

def splitRange(rangeIni,rangeEnd):
    start = pd.Timestamp(rangeIni)
    end = pd.Timestamp(rangeEnd)
    parts = list(pd.date_range(start, end, freq='M')) 

    if start != parts[0]:
        parts.insert(0, start)
    if end != parts[-1]:
        parts.append(end)

    parts[0] -= pd.Timedelta('1d')
    pairs = zip(map(lambda d: d + pd.Timedelta('1d'), parts[:-1]), parts[1:]) #Slice last row for convenience, and first row for make the ranges, and zip it
    dfDateRanges = pd.DataFrame(pairs, columns = ['ini', 'end'])
    # pairs_str = list(map(lambda t: t[0].strftime('%Y-%m-%d') + ' - ' + t[1].strftime('%Y-%m-%d'), pairs))
    return dfDateRanges

def scalator(df,unDesiredColumns):
    column_scaler = {}
    for columnName in df.columns:
        if columnName not in unDesiredColumns:
            scaler = preprocessing.MinMaxScaler()
            df[columnName] = scaler.fit_transform(np.expand_dims(df[columnName].values, axis=1))
            column_scaler[columnName] = scaler
        else:
            pass

def shuffle_in_unison(a, b):
    # shuffle two arrays in the same way
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)

def get_dataYahoo(ticker, scaled = True, dropTicker = False, news = True, shuffle = True, period = 0, interval = None, rangeIni = False, rangeEnd = False):
    """Method for the data extraction of the selected ticker with modified results based on different parameters

    Args:
        ticker (string_or_pd.DataFrame): Selected ticker for data extraction
        scaled (bool, optional): Scale values 0 to 1 for better performance. Defaults to True.
        dropTicker (bool, optional): Remove ticker column from dataframe. Defaults to False.
        news (bool, optional): Adquire news for training. Defaults to True.
        shuffle (bool, optional): Shuffle the data or not. Defaults to True.
        period (int, optional): Period of data search (0 week, 1 month, 2 year, 3 all). Defaults to 1.
        interval (_type_, optional): Interval of the data (weekly, monthly). Defaults to None.

    Raises:
        TypeError: The type of the ticker is not string

    Returns:
        pd.DataFrame: Dataframe with the info of the ticker selected with the parameters selected
    """
    # <!------------------- Extraction ---------------------->
    # http://theautomatic.net/yahoo_fin-documentation/#methods
    #           
    if isinstance(ticker, str):
        if rangeIni and rangeEnd:
            df = si.get_data(ticker, start_date = rangeIni , end_date = rangeEnd)
        elif period != 3:
            endDateRange = date.today()
            if period == 0:
                startDateRange = endDateRange - pd.DateOffset(months=1)
            elif period == 1:
                startDateRange = endDateRange - pd.DateOffset(months=6)
            elif period == 2:
                startDateRange = endDateRange - pd.DateOffset(years=1)
            df = si.get_data(ticker, start_date = startDateRange , end_date = endDateRange)
        else:
            df = si.get_data(ticker, interval)
    elif isinstance(ticker, pd.DataFrame):
        df = ticker
    else:
        raise TypeError("Type is not 'str' or 'pd.DataFrame'")
    
    # <!------------------- Manipulation ---------------------->
    # Create column date instead of using it as index
    if "date" not in df.columns:
        df["date"] = df.index
        # Reset index
        df = df.reset_index(drop=True)
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')

    # Delete duplicated rows (yahoo fin sometimes gives data of the same day twice)
    df = df.drop_duplicates(subset=["date"], keep=False)

    # Drop Ticker
    if dropTicker:
        del df["ticker"]

    # Scale values 0-1 (better perfomance)
    if scaled:
        scalator(df)

    # Add news (deberia de sacar positividad de las noticias medias en referencia al tema para agregarlo como positividad de las noticias como linea adicional)
    if news:
        pass

    # Shuffle data
    if shuffle:
        pass

    return df

def lWCFix(df):
    """Method to make the yahoo fin data from the "get_dataYahoo" method easier and faster to put in the chart

    Args:
        df (pd.DataFrame): The dataframe where info is going to get generated

    Returns:
        pd.DataFrame: Dataframes with the data splitted on two variables one for candle chart and the other for the histogram
    """
    # Fix for lightweightcharts lib
    df = df.rename(columns={"date":"time"})
    df = df.rename(columns={"volume":"value"})
    # Create color column for volume series
    df['color'] = np.where(df['open'] < df['close'] ,'rgba(0, 150, 136, 0.8)' , 'rgba(255,82,82, 0.8)')
    # Split the dataframe into 2 dataframes, 1 with volume data, and other with the rest of the data
    candleData = df.drop(columns=['value', 'adjclose', 'color'])
    volumeData = df.filter(['value', 'time', 'color'])
    # Save as dict for passing to JS
    candleData = candleData.to_dict(orient='records')
    volumeData = volumeData.to_dict(orient='records')
    return candleData, volumeData

def newsChecker(sbl):
    dateToday = date.today()
    dateDaysAgo = dateToday - pd.DateOffset(days=7)
    news = New.objects.filter(ticker=sbl).filter(date__gt=dateDaysAgo)
    if len(news) < 3:
        listReturn = newsExtract(sbl, dateToday, dateDaysAgo, save = True)
    else:
        listReturn = New.objects.filter(ticker=sbl).filter(pk__gte=New.objects.count() - 3)
    return listReturn

def newsExtract(sbl, iniRange, endRange, provider = False, all = False, numberOfNews = 3, save = False):
    """news extractor

    Args:
        sbl (str): search param
        iniRange (_type_): begining of range
        endRange (_type_): end of rnage
        provider (bool, optional): _description_. Defaults to False.
        all (bool, optional): all news. Defaults to False.
        numberOfNews (int, optional): number of news, if not all. Defaults to 3.
        save (bool, optional): save news. Defaults to False.

    Returns:
        list: list of news
    """    
    iniRange = iniRange.strftime('%m-%d-%Y')
    endRange = endRange.strftime('%m-%d-%Y')

    googlenews = GoogleNews(start=iniRange,end=endRange, lang='en')

    googlenews.search(sbl)
    listNews = googlenews.results()

    listReturn = []
    if all:
        numberOfNews = len(listNews)
        print(numberOfNews)
    for index in range(numberOfNews):
        urlParsed = urlparse(listNews[index]['link'])
        provider = re.sub('www.', '',urlParsed.netloc)
        provider = re.sub('\..*', '',provider)
        # listReturn.append(New(title = listNews[index]['title'],date = listNews[index]['datetime'],desc = listNews[index]['desc'],link = listNews[index]['link'],provider = provider))
        if listNews[index]['datetime']:
            listReturn.append([listNews[index]['title'],listNews[index]['datetime'].date(),listNews[index]['desc'],listNews[index]['link'],provider])  
        # dfNews = pd.DataFrame(listReturn, columns=["title", "date", "desc", "link", "provider"])
    if save:
        model_instances = [ New(
            title = new[0],
            date = new[1],
            desc = new[2],
            link = new[3],
            provider = new[4],
            ticker = sbl
        ) for new in listReturn ]
        New.objects.bulk_create(model_instances)
    return listReturn

def addIndicators(df, BB = False, DEMA = False, RSI = False, MACD = False):
    """https://mrjbq7.github.io/ta-lib/

    Args:
        df (_type_): _description_
        BB (bool, optional): _description_. Defaults to False.
        DEMA (bool, optional): _description_. Defaults to False.
        RSI (bool, optional): _description_. Defaults to False.
        MACD (bool, optional): _description_. Defaults to False.
    """
    if BB:
        df['upperband'], df['middleband'], df['lowerband'] = BBANDS(df['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    
    # Clean
    df = df.dropna()
    df = df.reset_index(drop=True)
    return df
    

def deep_learning_model_creation(X_Train, layers = 2, optimizer="rmsprop", units=256, dropout=0.2, loss="mean_absolute_error"):
    # https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233
    model = Sequential()
    for i in range(layers):
        if i == 0:
            model.add(LSTM(units=units,return_sequences=True,input_shape=(X_Train.shape[1], 1)))
        elif i == layers - 1:
            model.add(LSTM(units=units,return_sequences=False))
        else:
            model.add(LSTM(units=units,return_sequences=True))
        # Add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer=optimizer,loss=loss)
    return model


def ml_launch(df, epochs = 100, batch_size = 32, type=0, shuffle = False): #https://www.youtube.com/watch?v=6_2hzRopPbQ
    # poner el lookout step como la cantidad de dias que se quiere mirar en el futuro, poner una linea y a tomar por culo
    
    df = df.assign(future=df['adjclose'].shift(-1))
    df.dropna(subset=['future'], how='all', inplace=True)
    # df['rising'] = df.apply(lambda x : 1 if x['future'] >= x['adjclose'] else 0, axis=1)
    df = df.drop(columns=['date'])

    X_Train, X_Test, Y_Train, Y_Test = train_test_split(df.drop(['future'], axis=1), df['future'], test_size=0.2, random_state=1, shuffle = shuffle)
    
    # Deep learning
    # https://towardsdatascience.com/a-quick-deep-learning-recipe-time-series-forecasting-with-keras-in-python-f759923ba64
    if type == 1:
        model = deep_learning_model_creation(X_Train, optimizer = "adam", loss = "huber_loss", dropout=0.4)
        model.fit(
            X_Train,
            Y_Train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_Test, Y_Test)
        )

    # kNN
    # https://towardsdatascience.com/forecasting-of-periodic-events-with-ml-5081db493c46
    if type == 2:        
        params = {'n_neighbors':[2,3,4]}
        knn = KNeighborsRegressor(algorithm = 'auto', weights = 'distance')
        model = GridSearchCV(knn, params, cv=5)
        model.fit(
            X_Train,
            Y_Train
        )
    
    y_pred = model.predict(X_Test)
    error = sqrt(mean_squared_error(Y_Test,y_pred))
    print(error)
    
    return model
    