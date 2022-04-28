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


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import deque
import os


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

# def scalator(df,unDesiredColumns):
#     column_scaler = {}
#     for columnName in df.columns:
#         if columnName not in unDesiredColumns:
#             scaler = preprocessing.MinMaxScaler()
#             df[columnName] = scaler.fit_transform(np.expand_dims(df[columnName].values, axis=1))
#             column_scaler[columnName] = scaler
#         else:
#             pass

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

    # # Scale values 0-1 (better perfomance)
    # if scaled:
    #     scalator(df)

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
    news = New.objects.filter(date__gt=dateDaysAgo)
    if len(news) < 3:
        listReturn = newsExtract(sbl, dateToday, dateDaysAgo, save = True)
    else:
        listReturn = New.objects.filter(pk__gte=New.objects.count() - 3)
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
            provider = new[4]
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
    

def model_creation(sequence_length, n_features, units=256, cell=LSTM, n_layers=2, dropout=0.3,
                loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False ):
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # first layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True), batch_input_shape=(None, sequence_length, n_features)))
            else:
                model.add(cell(units, return_sequences=True, batch_input_shape=(None, sequence_length, n_features)))
        elif i == n_layers - 1:
            # last layer
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=False)))
            else:
                model.add(cell(units, return_sequences=False))
        else:
            # hidden layers
            if bidirectional:
                model.add(Bidirectional(cell(units, return_sequences=True)))
            else:
                model.add(cell(units, return_sequences=True))
        # add dropout after each layer
        model.add(Dropout(dropout))
    model.add(Dense(1, activation="linear"))
    model.compile(loss=loss, metrics=["accuracy"], optimizer=optimizer)
    return model


def learning_launch(df, epochs = 200, batch_size = 32): #https://www.youtube.com/watch?v=6_2hzRopPbQ
    n_steps=2
    shuffle=True
    lookup_step=1
    split_by_date=True
    test_size=0.2
    feature_columns=['adjclose', 'volume', 'open', 'high', 'low']
    scale = True
    

    # this will contain all the elements we want to return from this function
    result = {}
    # we will also return the original dataframe itself
    result['df'] = df.copy()


    if scale:
        column_scaler = {}
        # scale the data (prices) from 0 to 1
        for column in feature_columns:
            scaler = preprocessing.MinMaxScaler()
            df[column] = scaler.fit_transform(np.expand_dims(df[column].values, axis=1))
            column_scaler[column] = scaler
        # add the MinMaxScaler instances to the result returned
        result["column_scaler"] = column_scaler


    df['future'] = df['adjclose'].shift(-lookup_step)

    last_sequence = np.array(df[feature_columns].tail(lookup_step))
    df.dropna(inplace=True)

    sequence_data = []
    sequences = deque(maxlen=n_steps)
    for entry, target in zip(df[feature_columns + ["date"]].values, df['future'].values):
        sequences.append(entry)
        if len(sequences) == n_steps:
            sequence_data.append([np.array(sequences), target])

    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    result['last_sequence'] = last_sequence

    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)

    X = np.array(X)
    y = np.array(y)
    if split_by_date:
        # split the dataset into training & testing sets by date (not randomly splitting)
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"]  = X[train_samples:]
        result["y_test"]  = y[train_samples:]
        if shuffle:
            # shuffle the datasets for training (if shuffle parameter is set)
            shuffle_in_unison(result["X_train"], result["y_train"])
            shuffle_in_unison(result["X_test"], result["y_test"])
    else:    
        # split the dataset randomly
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y, 
                                                                                test_size=test_size, shuffle=shuffle)

    # get the list of test set dates
    dates = result["X_test"][:, -1, -1]
    # retrieve test features from the original dataframe
    result["test_df"] = result["df"].reindex(dates, axis = 1)
    # # remove duplicated dates in the testing dataframe
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep='first')]
    # # remove dates from the training/testing sets & convert to float32
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)


    model = model_creation(n_steps, len(feature_columns), loss='huber_loss', units=256, cell=LSTM, n_layers=2,
                    dropout=0.4, optimizer='adam')


    history = model.fit(result["X_train"], result["y_train"],
                        batch_size=64,
                        epochs=epochs,
                        validation_data=(result["X_test"], result["y_test"]),
                        verbose=1)

    pass
    