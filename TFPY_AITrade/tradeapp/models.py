from django.db import models


#crea modelos de noticias, extrae noticias y sacalas

# Create your models here.
class New(models.Model):
    title = models.CharField(max_length=100)
    date = models.DateField()
    desc = models.CharField(max_length=400)
    link = models.CharField(max_length=200)
    provider = models.CharField(max_length=50)

    def __str__(self):
        return self.title