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

def symbol_view(request, sbl):
    tickerData = get_dataYahoo(sbl, scaled = False, dropTicker = True, period = 3)
    # Swap rows to columns
    # tickerData = tickerData.transpose()
    # # JS doesn't read pandas so we need to make it a JSON object
    # tickerData = tickerData.to_json()
    print(tickerData)
    dictPar = {
        'sbl':sbl,
        'tickerData':tickerData,
    }
    return render(request, 'tradeapp/symbol.html', dictPar)




# class MarketListView(ListView):
#     template_name = 'tradeapp/markets.html'

# class SymbolDetailView(DetailView):
#     template_name = 'tradeapp/symbol.html'