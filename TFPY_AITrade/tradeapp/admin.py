from django.contrib import admin
from .models import New, AiModel, Ticker

# Register your models here.
admin.site.register(New)
admin.site.register(AiModel)
admin.site.register(Ticker)