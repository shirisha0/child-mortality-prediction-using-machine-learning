from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)

class child_mortality_type(models.Model):


    country= models.CharField(max_length=3000)
    start_date= models.CharField(max_length=3000)
    end_date= models.CharField(max_length=3000)
    days= models.CharField(max_length=3000)
    year= models.CharField(max_length=3000)
    week= models.CharField(max_length=3000)
    total_deaths= models.CharField(max_length=3000)
    Child_Age= models.CharField(max_length=3000)
    Disease= models.CharField(max_length=3000)
    Medicine_Status= models.CharField(max_length=3000)
    Life_expectancy= models.CharField(max_length=3000)
    RID= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)
