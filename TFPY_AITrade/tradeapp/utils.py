# https://www.youtube.com/watch?v=jrT6NiM46jk matplotlib (not needed but maybe usefull in the future)
from yahoo_fin import stock_info as si
from sklearn import preprocessing
import numpy as np
import pandas as pd
from datetime import date
from GoogleNews import GoogleNews
from urllib.parse import urlparse
import re
from talib import BBANDS

from .models import New


def get_dataYahoo(ticker, scaled = True, dropTicker = False, news = True, shuffle = True, period = 0, interval = None):
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
    # Type check
    if isinstance(ticker, str):
        # Get date range 
        endDateRange = date.today()
        if period != 3:
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

def newsExtract(sbl, provider = False):
    """Method for extracting news from a ticker and save or load them to/from the database as well as returning them

    Args:
        sbl (_type_): symbol name
        provider (bool, optional): We only need provider when showing news, when getting news for prediction the provider is not needed. Defaults to False.

    Returns:
        list: list of the lastest 3 news
    """
    dateToday = date.today()
    dateDaysAgo = dateToday - pd.DateOffset(days=7)
    news = New.objects.filter(date__gt=dateDaysAgo)
    
    startRange = dateToday.strftime('%m-%d-%Y')
    endRange = dateDaysAgo.strftime('%m-%d-%Y')

    if len(news) < 3:
        googlenews = GoogleNews(start=startRange,end=endRange)
        googlenews.search(sbl)
        listNews = googlenews.results()
        for index in range(3):
            urlParsed = urlparse(listNews[index]['link'])
            provider = re.sub('www.', '',urlParsed.netloc)
            provider = re.sub('\..*', '',provider)
            newNew = New(title = listNews[index]['title'], date = listNews[index]['datetime'], desc = listNews[index]['desc'], link = listNews[index]['link'], provider = provider)
            newNew.save()

    listReturn = New.objects.filter(pk__gte=New.objects.count() - 3)
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
    



        

    