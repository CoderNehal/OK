# dataminingapp/models.py
from django.db import models

class Data(models.Model):
    file = models.FileField(upload_to='uploads/')
