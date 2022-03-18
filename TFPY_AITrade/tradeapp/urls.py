# https://docs.djangoproject.com/en/4.0/topics/http/urls/

from django.urls import path
from .views import (
    home_view,
    markets_view,
    symbol_view,
    # MarketListView,
    # SymbolDetailView,
)

app_name = 'tradeapp'

urlpatterns = [
    path('', home_view, name='home'),
    path('markets/', markets_view, name='markets'), # path('markets/', MarketListView.as_view(), name='market'),
    path('markets/<sbl>/', symbol_view, name='symbol'), # path('markets/<pk>/', SymbolDetailView.as_view(), name='symbol'),   
]