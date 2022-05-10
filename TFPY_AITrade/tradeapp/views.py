from fileinput import filename
from re import X
from typing import NewType
from django.http import JsonResponse
from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.http import HttpResponseRedirect
from django.http import HttpResponse
from .utils import get_dataYahoo, lWCFix, newsChecker, newsExtract, addIndicators, splitRange, scalator, ml_launch
from .models import AiModel
from django.conf import settings


from textblob import TextBlob
import pickle
import os



import numpy as np
import pandas as pd
from datetime import datetime


import timeit

# Create your views here.
def home_view(request):
    hello = 'hello world from the view'
    return render(request, 'tradeapp/home.html', {'h':hello})

def markets_view(request):
    return render(request, 'tradeapp/markets.html', {})

def symbol_view(request, sbl):
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

    latestNews = newsChecker(sbl)
    
    indicators = pd.DataFrame()
    indicators['full'] = ['Bollinger Bands', 'Double Exponential Moving Average', 'Relative Strength Index', 'Moving Average Convergence Divergence']
    indicators['acronym'] = ['BB', 'DEMA', 'RSI','MACD']

    algorithms = ['Neural network', 'K-NN']


    models = AiModel.objects.filter(ticker=sbl)

    data = {
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

    print(otherSelection)
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
        print(algorithm)
        if(algorithm > 0):
            rangeIni = datetime.strptime(request.POST.get('rangeIni'), '%Y-%m-%d')
            rangeEnd = datetime.strptime(request.POST.get('rangeEnd'), '%Y-%m-%d')
            scalate = request.POST.get('scalate')
            benchmark = request.POST.get('benchmark')
            modelName = request.POST.get('model')
            modelDesc = request.POST.get('description')



            BB = request.POST.get('BB')
            DEMA = request.POST.get('DEMA')
            RSI = request.POST.get('RSI')
            MACD = request.POST.get('MACD')

            
            df = get_dataYahoo(ticker = benchmark, scaled = False, dropTicker = True, rangeIni = rangeIni, rangeEnd = rangeEnd)
            # Add indicators
            df = addIndicators(df, BB = BB)
            # Add news if news
            if (request.POST.get('news')):
                dfDateRanges = splitRange(rangeIni,rangeEnd)

                listForDF = []
                # ESTO HAY QUE MEJORARLO ITER ROWS ES MALA IDEA PERO NO SE COMO HACERLO AHORA MISMO
                for index, r in dfDateRanges.iterrows():
                    listForDF.extend(newsExtract(benchmark,r['ini'],r['end'], all = True))
                dfNewsPLN = pd.DataFrame(listForDF, columns=["title", "date", "desc", "link", "provider"])

                # Make PLN of description
                dfNewsPLN['polarity'] = dfNewsPLN['desc'].apply(lambda x : TextBlob(x).sentiment.polarity)
                dfNewsPLN['subjectivity'] = dfNewsPLN['desc'].apply(lambda x : TextBlob(x).sentiment.subjectivity)

                # Drop useless columns
                dfNewsPLN = dfNewsPLN.drop(columns=['title', 'desc', 'link', 'provider'])

                # Do a mean of values
                dfNewsPLN = dfNewsPLN.groupby('date', as_index = False).mean()

                # Make date same type for left join
                df['date'] = pd.to_datetime(df['date'])
                dfNewsPLN['date'] = pd.to_datetime(dfNewsPLN['date'])

                # Do left join
                df = df.merge(dfNewsPLN, on=['date'], how="left")

                # Fill nan values with latest values
                df = df.ffill()

                # Drop NaN values
                df.dropna(subset=['polarity'], how='all', inplace=True)
            # Scalate results
            if scalate:
                scalator(df,['date', 'polarity', 'subjectivity'])

            # https://stackoverflow.com/questions/46830427/save-classifier-to-postrgesql-database-in-scikit-learn
            model = ml_launch(df, type = algorithm, epochs=100, batch_size = 3)

            # El modelo puede tener nombres repetidos, eso = bug mirarlo antes, 
            filename =  str(modelName) + ".h5"
            modelPath = os.path.join(settings.MEDIA_ROOT, 'models', filename)
            model.save(modelPath)
            # https://keras.io/getting_started/faq/#what-are-my-options-for-saving-models
            # pickledModel = pickle.dumps(model)
            newModel = AiModel(
                name = modelName,
                desc = modelDesc,
                ticker = benchmark,
                model = modelPath,
                scaled = scalate,
                BB = BB,
                DEMA = DEMA,
                RSI = RSI,
                MACD = MACD
                )
            newModel.save()
            
        else:
            print('no selected dar error')
        
    return HttpResponse('')

def predict(request, sbl):
    if request.GET.get('action') == 'predict':
        benchmark = request.GET.get('benchmark')
        idModel = request.GET.get('modelSelected')
        model = AiModel.objects.filter(id=idModel)
        print(model[0].desc)
        # tengo el modelo
        # hay que analizar el modelo (scaled, bb, etc..)
        # sacar el ultimo dia de datos del ticker seleccionado
        #     (mirar si podemos tomar en cuento los dias anteriores tambien)
        # hacer predict
    return HttpResponse('')