from fileinput import filename
from re import X
from typing import NewType
from django.http import JsonResponse
from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from .utils import get_dataYahoo, lWCFix, newsChecker, newsExtract, addIndicators, scalator, ml_launch, newsPLNFitDF
from .models import AiModel, New, Ticker
from django.conf import settings



import os
from tensorflow.keras.models import load_model
from pickle import dump, load #https://machinelearningmastery.com/how-to-save-and-load-models-and-data-preparation-in-scikit-learn-for-later-use/
import json

import time

import numpy as np
import pandas as pd
from datetime import datetime


import timeit

# Create your views here.
def home_view(request):
    tickers = Ticker.objects.all()
    gainers = si.get_day_gainers(count=7)
    losers = si.get_day_losers(count=7)
    most_active = si.get_day_most_active(count=7)

    
    # Download latest news from db
    latestNews = New.objects.order_by('-date')[:8][::-1]
    # If there are no news stored get news from the most active symbol
    if not latestNews:
        latestNews = newsChecker(gainers['Symbol'][:1].tolist(), quant_news = 6) 

    data = {
        'tickers' : tickers,
        'gainers' : gainers,
        'losers' : losers,
        'most_active' : most_active,
        'latestNews' : latestNews
    }

    return render(request, 'tradeapp/home.html', data)

def markets_view(request):
    # tickerspool1 = si.tickers_sp500()
    # tickerspool2 = si.tickers_nasdaq()
    # tickerspool3 = si.tickers_dow()
    # tickerspool4 = si.tickers_other()
    # framesToApend = [tickerspool1, tickerspool2, tickerspool3]

    # allTickers = pd.DataFrame()
    # allTickers = []
    # for df in framesToApend:
    #     allTickers.extend(df)

    # result = [] 
    # [result.append(x) for x in allTickers if x not in result] 
    #     allTickers = pd.concat([allTickers, df])
    
    # allTickers = allTickers.filter([:])

    # for ti in result:
    #     tick = Ticker(symbol=ti)
    #     tick.save()


    tickers = Ticker.objects.all()
    most_active = si.get_day_most_active()
    data = {
        'tickers' : tickers,
        'most_active' : most_active
    }

    return render(request, 'tradeapp/markets.html', data)

def symbol_view(request, sbl):
    tickers = Ticker.objects.all()
    # Extract data
    tickerData = get_dataYahoo(sbl, scaled = False, dropTicker = True, period = 0)

    # LightweightCharts Fix
    candleData, volumeData = lWCFix(tickerData)

    actualSeletion = 'Last month'
    
    otherSelection = [
        "Last 6 months",
        "Last year",
        "Full"
    ]

    latestNews = newsChecker([sbl])
    indicators = pd.DataFrame()
    indicators['full'] = ['Bollinger Bands', 'Double Exponential Moving Average', 'Relative Strength Index', 'Moving Average Convergence Divergence']
    indicators['acronym'] = ['BB', 'DEMA', 'RSI','MACD']

    algorithms = ['Neural network', 'K-NN']


    models = AiModel.objects.filter(ticker=sbl)

    data = {
        'tickers' : tickers,
        'sbl' : sbl,
        'candleData' : candleData,
        'volumeData' : volumeData,
        'actualSelection' : actualSeletion,
        'otherSelection' : otherSelection,
        'latestNews' : latestNews,
        'indicators' : indicators,
        'algorithms' : algorithms,
        'models' : models
    }

    return render(request, 'tradeapp/symbol.html', data)

def answer(request):
    sbl = request.GET.get('sbl')
    period = request.GET.get('period')
    actualSeletion = period
    if (period == 'Last month'):
        period = 0
    elif (period == 'Last 6 months'):
        period = 1
    elif (period == 'Last year'):
        period = 2
    else:
        period = 3

    # Create a dataframe with all the options the filter out the current one
    otherSelection = np.array(['Last month', 'Last 6 months', 'Last year', 'Full'])
    otherSelection = np.delete(otherSelection, np.argwhere(otherSelection == actualSeletion))
    otherSelection = pd.DataFrame(data = otherSelection, columns = ['value'])
    otherSelection = otherSelection.to_dict(orient='records')

    # Extract data
    tickerData = get_dataYahoo(sbl, scaled = False, dropTicker = True, period = period)
    # LightweightCharts Fix
    candleData, volumeData = lWCFix(tickerData)

    data = {
        'candleData':candleData,
        'volumeData':volumeData,
        'actualSelection':actualSeletion,
        'otherSelection':otherSelection
    }
    return JsonResponse(data)

def createModel(request, sbl):
    if request.POST.get('action') == 'create-model':
        algorithm = int(request.POST.get('algorithm'))
        if(algorithm > 0):
            # Get POST data
            rangeIni = datetime.strptime(request.POST.get('rangeIni'), '%Y-%m-%d')
            rangeEnd = datetime.strptime(request.POST.get('rangeEnd'), '%Y-%m-%d')
            lookup = int(request.POST.get('lookup'))
            scalate = request.POST.get('scalate')
            benchmark = request.POST.get('benchmark')
            modelName = request.POST.get('model')
            modelDesc = request.POST.get('description')
            BB = request.POST.get('BB')
            DEMA = request.POST.get('DEMA')
            RSI = request.POST.get('RSI')
            MACD = request.POST.get('MACD')

            # Get stock data
            df = get_dataYahoo(ticker = benchmark, dropTicker = True, rangeIni = rangeIni, rangeEnd = rangeEnd)

            # Add indicators
            df = addIndicators(df, BB = BB)

            # Add news if news
            if (request.POST.get('news')):
                df = newsPLNFitDF(df, benchmark, rangeIni, rangeEnd)

            # Scalate results
            if scalate:                
                scaler = scalator(df,['date', 'polarity', 'subjectivity'])['adjclose'] # Save scaler of adjclose to inverse in future

            # Check if data can fit the lookup
            if(len(df) >= (lookup * 2)): #Half the data is lost as NaNs since there's no prediction possible for them, we need double the data
                model = ml_launch(df, lookup = lookup, type = algorithm, epochs=100, batch_size = 3)
                # !!!!!!!!!!El modelo puede tener nombres repetidos, eso = bug mirarlo antes!!!!!!!!!!!!!!!
                # Model save and load to database
                filename =  str(modelName)

                # Neural networks save differently
                if (algorithm == 1):
                    modelPath = os.path.join(settings.MEDIA_ROOT, 'models', filename + ".h5")
                    model.save(modelPath)
                    keras = True
                else:
                    modelPath = os.path.join(settings.MEDIA_ROOT, 'models', filename + ".pkl")
                    dump(model, open(modelPath, 'wb'))
                    keras = False
                
                # Save scaler for future inversion
                scalerPath = os.path.join(settings.MEDIA_ROOT, 'scalers', filename + ".pkl")
                dump(scaler, open(scalerPath, 'wb'))
                newModel = AiModel(
                    name = modelName,
                    desc = modelDesc,
                    ticker = benchmark,
                    lookup = lookup,
                    model = modelPath,
                    scaler = scalerPath,
                    keras = keras,
                    scaled = scalate,
                    BB = BB,
                    DEMA = DEMA,
                    RSI = RSI,
                    MACD = MACD
                    )
                newModel.save()
            else:
                print('There is too low data for this lookup step')  
        else:
            print('No algorithm selected')
    return HttpResponse('')

def predict(request):
    if request.GET.get('action') == 'predict':
        # Get GET data
        benchmark = request.GET.get('benchmark')
        idModel = request.GET.get('modelSelected')

        # Get model from DB
        model = AiModel.objects.filter(id=idModel)

        # Extract data based on the lookup (we need last values, equal in number to lookup, more data is useless)
        df = get_dataYahoo(ticker = benchmark, dropTicker = True, period = 4, lookup = model[0].lookup)

        # Add indicators if model needs them
        df = addIndicators(df, BB = model[0].BB)

        # Add news if the model needs news
        if model[0].news:
            df = newsPLNFitDF(df, benchmark, df['date'].min(), df['date'].max())

        # Create result dataframe
        dfPred = df.iloc[-1:]
        dfPred = dfPred.filter(['adjclose', 'date'])
        dfPred = dfPred.rename(columns={"date":"time"})
        dfPred = dfPred.rename(columns={"adjclose":"value"})

        # Erase useless columns
        df = df.drop(columns=['date']) 

        # Scalate results        
        if model[0].scaled:
            scalator(df,['polarity', 'subjectivity'])
            # Load scaler      
            scaler = load(open(str(model[0].scaler), 'rb'))

        # Load model
        if(model[0].keras):
            loadedModel = load_model(str(model[0].model))  
        else:
            loadedModel = load(open(str(model[0].model), 'rb'))

        # Predict and inverse result
        prediction = loadedModel.predict(df.iloc[-model[0].lookup:])

        # Scale model if is required
        if model[0].scaled:
            if not model[0].keras:
                prediction = np.array(prediction).reshape(len(prediction),1) # The size of the prediction must be shape [[1],[2]]
            inversed = scaler.inverse_transform(prediction)
            prediction = inversed

        # Fix type
        dfPred['time'] = pd.to_datetime(dfPred['time'])

        # Add to the pred dataframe the predictions with summed date values
        # https://stackoverflow.com/questions/72247655/extend-a-dataframe-with-values-and-next-dates/
        df1 = pd.DataFrame({'value': np.array(prediction).flatten(),
                            'time': pd.date_range(dfPred['time'].max(), periods=len(prediction)+1, 
                                                freq='D', inclusive='right')})
        out = pd.concat([dfPred, df1], ignore_index=True)

        # Fix for java
        out['time'] = out['time'] + pd.DateOffset(hours=12) # Java needs hours above the 00 to be able to detect it as a next day

        # Make the data JSON
        prediction = out.to_dict(orient='records')
        data = {
            'prediction' : prediction,
        }
    return JsonResponse(data)