# https://docs.djangoproject.com/en/4.0/topics/http/urls/

from django.urls import path
from .views import (
    home_view,
    markets_view,
    symbol_view,
    # MarketListView,
    # SymbolDetailView,
    answer
)

app_name = 'tradeapp'

urlpatterns = [
    path('', home_view, name='home'),
    path('markets/', markets_view, name='markets'),
    path('markets/<sbl>/', symbol_view, name='symbolWeek'), 
    path('ajax/get_response/', answer, name='get_response'),
]