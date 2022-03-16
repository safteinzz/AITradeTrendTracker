from distutils.command.upload import upload
from django.db import models

# Create your models here.
class User(models.Model):
    alias = models.CharField(max_length=120)
    avatar = models.ImageField(upload_to='users', default='no_avatar.png')

    # String representation
    def __str__(self):
        return str(self.alias)
