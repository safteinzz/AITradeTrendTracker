from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from .utils import get_dataYahoo


# Create your views here.
def home_view(request):
    hello = 'hello world from the view'
    return render(request, 'tradeapp/home.html', {'h':hello}) #lo ultimo es un diccionario que le llega al html

def markets_view(request):
    return render(request, 'tradeapp/markets.html', {})

def symbol_view(request, sbl):
    tickerData = get_dataYahoo(sbl, scaled = False, dropTicker = True) #si falla tiene que dar 404    https://docs.djangoproject.com/en/4.0/ref/urls/
    print(tickerData)
    return render(request, 'tradeapp/symbol.html', {'sbl':sbl})




# class MarketListView(ListView):
#     template_name = 'tradeapp/markets.html'

# class SymbolDetailView(DetailView):
#     template_name = 'tradeapp/symbol.html'