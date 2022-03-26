from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .utils import get_dataYahoo


# Create your views here.
def home_view(request):
    hello = 'hello world from the view'
    return render(request, 'tradeapp/home.html', {'h':hello})

def markets_view(request):
    return render(request, 'tradeapp/markets.html', {})

def symbol_view(request, sbl, pd = 'weekly'):
    range = 0
    if pd == 'monthly':
        range = 1
    elif pd == 'yearly':
        range = 2
    elif pd == 'full':
        range = 3

    tickerData = get_dataYahoo(sbl, scaled = False, dropTicker = True, period = 0)
    # Fix for lightweightcharts lib
    tickerData = tickerData.rename(columns={"date":"time"})
    # tickerData = tickerData.drop([tickerData.index[2957]])
    print("------------------------------------------")
    print (tickerData)
    print("------------------------------------------")
    # Save as dict for passing to JS
    data = tickerData.to_dict(orient='records')
    # print(data)
    dictPar = {
        'sbl':sbl,
        'data':data
    }
    return render(request, 'tradeapp/symbol.html', dictPar)




# class MarketListView(ListView):
#     template_name = 'tradeapp/markets.html'

# class SymbolDetailView(DetailView):
#     template_name = 'tradeapp/symbol.html'