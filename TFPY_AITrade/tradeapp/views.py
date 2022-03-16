from yahoo_fin import stock_info as si
from django.shortcuts import render
from django.views.generic import ListView, DetailView


# Create your views here.
def home_view(request):
    hello = 'hello world from the view'
    return render(request, 'tradeapp/home.html', {'h':hello}) #lo ultimo es un diccionario que le llega al html

def markets_view(request):
    return render(request, 'tradeapp/markets.html', {})

def symbol_view(request, sbl):
    ticketData = si.get_data(sbl) #si falla tiene que dar 404
    if ticketData:        
        return render(request, 'tradeapp/symbol.html', {'sbl':sbl})




# class MarketListView(ListView):
#     template_name = 'tradeapp/markets.html'

# class SymbolDetailView(DetailView):
#     template_name = 'tradeapp/symbol.html'