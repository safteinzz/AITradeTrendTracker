from django.db import models


#crea modelos de noticias, extrae noticias y sacalas

# Create your models here.
class New(models.Model):
    title = models.CharField(max_length=100)
    date = models.DateField(default=None, blank=True, null=True)
    desc = models.CharField(max_length=400)
    link = models.CharField(max_length=200)
    provider = models.CharField(max_length=50)
    ticker = models.CharField(max_length=50)

    def __str__(self):
        return self.title

class AiModel(models.Model):
    name = models.CharField(max_length=100, null=True)
    desc = models.CharField(max_length=400, null=True)
    ticker = models.CharField(max_length=50)
    # model = models.BinaryField()
    model = models.FileField(upload_to='models')
    scaled = models.BooleanField(default=False)
    BB = models.BooleanField(default=False)
    DEMA = models.BooleanField(default=False)
    RSI = models.BooleanField(default=False)
    MACD = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name