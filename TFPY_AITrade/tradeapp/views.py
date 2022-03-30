from django.http import JsonResponse
from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .utils import get_dataYahoo, LWCFix

import numpy as np
import pandas as pd

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
    candleData, volumeData = LWCFix(tickerData)

    actualSeletion = 'Last month'
    otherSelection = [
        "Last 6 months",
        "Last year",
        "Full"
    ]

    data = {
        'sbl':sbl,
        'candleData':candleData,
        'volumeData':volumeData,
        'actualSelection':actualSeletion,
        'otherSelection':otherSelection
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
    candleData, volumeData = LWCFix(tickerData)

    data = {
        'candleData':candleData,
        'volumeData':volumeData,
        'actualSelection':actualSeletion,
        'otherSelection':otherSelection
    }
    return JsonResponse(data)