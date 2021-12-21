from math import sin,cos,pi

import numpy as np
import pandas as pd

atts = ["X","Y","month","day","FFMC","DMC","DC","ISI","temp","RH","wind","rain","y"]
students = pd.read_csv('forestfires.csv', names=atts)
forestfire = students.drop("y", axis=1)

dict_days = {'sun':1, 'mon':2, 'tue':3, 'wed':4, 'thu':5, 'fri':6, 'sat':7}
dict_month = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}

#Calcula o sen e cos dos dias presentes na coluna "days"
def numerico_dias(dataset):
    sen_days = []
    cos_days = []
    days = dataset["day"]
    #Percorre a coluna "days"
    for day in days:
        #Gera o seno do day e guarda no vetor de senos
        sen_days.append(sin(2*pi*(dict_days[day]/7)))
        #Gera o cosseno do day e guarda no vetor de cossenos
        cos_days.append(cos(2*pi*(dict_days[day]/7)))
    print(np.array(sen_days))
    print()
    print(np.array(cos_days))
    print()

#Calcula o sen e cos dos dias presentes na coluna "month"
def numerico_mes(dataset):
    sen_month = []
    cos_month = []
    months = dataset["month"]
    for month in months:
        # Gera o seno do month e guarda no vetor de senos
        sen_month.append(sin(2*pi*(dict_month[month]/12)))
        # Gera o cosseno do month e guarda no vetor de cossenos
        cos_month.append(cos(2*pi*(dict_month[month]/12)))
    print(np.array(sen_month))
    print()
    print(np.array(cos_month))
    print()

numerico_dias(forestfire)
print("=======================================")
numerico_mes(forestfire)

