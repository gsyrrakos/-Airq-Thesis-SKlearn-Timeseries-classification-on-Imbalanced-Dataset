from datetime import datetime as dt
import pandas as pd
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.preprocessing import StandardScaler
import math

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
rate = 12

# ///////////////////
path4 = "F://sensor smoking//"

path5 = "F://sensor_normal//"

path6 = "F://sensor_cooking//"

pathf4 = "F://sensor smoking//Feutures//"
name4 = 'Feautures_AA_Smok_1.csv'

pathf5 = "F://sensor_normal//Feutures//"
name5 = 'Feautures_AA_normal_1.csv'

pathf6 = "F://sensor_cooking//Feutures//"
name6 = 'Feautures_AA_Cooking_1.csv'

# //////////////////




# paths edw allazw to value
path = "G://MyData//"

# ayto to path allzw

path2 = "G://MyData//smok//"
name2 = name5






# xroniko olisthisi gia 5 lepta
rateOfwindow = 30

glued_data = pd.DataFrame()
for file_name in glob.glob(path + '*.csv'):
    x = pd.read_csv(file_name, parse_dates=["TimeStamp"], index_col="TimeStamp")
    glued_data = pd.concat([glued_data, x], axis=0)
print(glued_data.head(35))

'''
df1 = pd.read_csv(path + name, parse_dates=["TimeStamp"], index_col="TimeStamp")
print(df1.index)


print(df1)

print(df1.index[0])
val=df1.index[0]
val1='2020-08-20 09:00:05'
'''
listwindow = list()
'''
sc = StandardScaler(with_mean=True, with_std=True)
sc.fit(glued_data[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']])
''''''
glued_data[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']] = sc.transform(
    glued_data[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']])
'''

# PM Sensors
def increaseRate(df1, val):
    rate = ((abs(df1[val].max()) - abs(df1[val].iloc[0]))) / abs(df1[val].iloc[0])
    return rate


# -------------------------------------------
def ChangeRate(df1, val):
    rate = abs(increaseRate(df1, val) - decreaseRate(df1, val))
    return rate


def ChangeMagn(df1, val):
    rate = abs(increaseMag(df1, val) - decreaseMag(df1, val))
    return rate


def ChangeIncreaseMagn_rate(df1, val):
    rate = abs(increaseRate(df1, val) - increaseMag(df1, val))
    return rate


def ChangeDecreaseMagn_rate(df1, val):
    rate = abs(decreaseRate(df1, val) - decreaseMag(df1, val))
    return rate


# --------------------------------------------


def increaseMag(df1, val):
    mag = (abs(df1[val].max()) - abs(df1[val].iloc[0]))
    return mag


def decreaseRate(df1, val):
    rate = ((abs(df1[val].max()) - abs(df1[val].iloc[-1])) / abs(df1[val].max()))
    print(abs(df1[val].iloc[-1]))
    print(abs(df1[val].max()))
    return rate


def decreaseMag(df1, val):
    mag = (abs(df1[val].max()) - abs(df1[val].iloc[-1]))
    return mag


def Std(df1, val):
    stdval = np.std(df1[val])
    return stdval


# -------------------------------------------------------------------------------------

def STDRATIOrate(df1, val, val2, val3):
    rate = abs(Std(df1, val) / (abs(Std(df1, val) + Std(df1, val2) + Std(df1, val3))))
    return rate


# --------------------------------------------------------------------------------------
# DHT sensor
def ChangeMagni(df1, val):
    mag = (abs(df1[val].max()) - abs(df1[val].min()))
    return mag


def NSTD(df1):
    dfA = df1
    dfc = df1[['PM 2.5']]
    df1['sum'] = dfc['PM 2.5'].apply(lambda x: abs(x - (dfc['PM 2.5'].mean())))

    nstd = ((1 / (abs(dfc['PM 2.5'].max()))) * math.sqrt((1 / 29) * ((df1['sum'].sum())) ** 2))

    return nstd



def func(listpdf):
    pds = listpdf

    # features
    Increase_RatePM1 = increaseRate(pds, 'PM 1')
    Increase_RatePM25 = increaseRate(pds, 'PM 2.5')
    Increase_RatePM10 = increaseRate(pds, 'PM 10')

    Increase_MagnitudePM1 = increaseMag(pds, 'PM 1')
    Increase_MagnitudePM25 = increaseMag(pds, 'PM 2.5')
    Increase_MagnitudePM10 = increaseMag(pds, 'PM 10')

    DeacreaseRatePM1 = decreaseRate(pds, 'PM 1')
    DeacreaseRatePM25 = decreaseRate(pds, 'PM 2.5')
    DeacreaseRatePM10 = decreaseRate(pds, 'PM 10')

    Deacrease_Magnitude1 = decreaseMag(pds, 'PM 1')
    Deacrease_Magnitude25 = decreaseMag(pds, 'PM 2.5')
    Deacrease_Magnitude10 = decreaseMag(pds, 'PM 10')
    # -----------------------------------------------------------------------

    ChangeRate1 = ChangeRate(pds, 'PM 1')
    ChangeRate25 = ChangeRate(pds, 'PM 2.5')
    ChangeRate10 = ChangeRate(pds, 'PM 10')

    ChangeMagn1 = ChangeMagn(pds, 'PM 1')
    ChangeMagn25 = ChangeMagn(pds, 'PM 2.5')
    ChangeMagn10 = ChangeMagn(pds, 'PM 10')

    ChangeIncreaseMagn_rate1 = ChangeIncreaseMagn_rate(pds, 'PM 1')
    ChangeIncreaseMagn_rate25 = ChangeIncreaseMagn_rate(pds, 'PM 2.5')
    ChangeIncreaseMagn_rate10 = ChangeIncreaseMagn_rate(pds, 'PM 10')

    ChangeDecreaseMagn_rate1 = ChangeDecreaseMagn_rate(pds, 'PM 1')
    ChangeDecreaseMagn_rate25 = ChangeDecreaseMagn_rate(pds, 'PM 2.5')
    ChangeDecreaseMagn_rate10 = ChangeDecreaseMagn_rate(pds, 'PM 10')

    STDRATIOrate1 = STDRATIOrate(pds, 'PM 1', 'PM 2.5', 'PM 10')
    STDRATIOrate25 = STDRATIOrate(pds, 'PM 1', 'PM 2.5', 'PM 10')
    STDRATIOrate10 = STDRATIOrate(pds, 'PM 1', 'PM 2.5', 'PM 10')

    STDPM1 = Std(pds, 'PM 1')
    STDPM25 = Std(pds, 'PM 2.5')
    STDPM10 = Std(pds, 'PM 10')
    STDHumidity = Std(pds, 'Humidity')
    STDTemp = Std(pds, 'Temperature')
    NSTDPM2 = NSTD(pds)

    CMagHUmidity = ChangeMagni(pds, 'Humidity')
    CMagTemperature = ChangeMagni(pds, 'Temperature')
    Feautures_AA_Smok_1 = pd.read_csv(path2 + name2)
    Feautures_AA_Smok_1 = Feautures_AA_Smok_1.drop(columns=['Unnamed: 0', ])
    Feautures_AA_Smok_11 = [Increase_RatePM1, Increase_RatePM25, Increase_RatePM10,
                            Increase_MagnitudePM1,
                            Increase_MagnitudePM25, Increase_MagnitudePM10,
                            DeacreaseRatePM1, DeacreaseRatePM25,
                            DeacreaseRatePM10, Deacrease_Magnitude1,
                            Deacrease_Magnitude25, Deacrease_Magnitude10, STDPM1,
                            STDPM25, STDPM10, STDHumidity, STDTemp, CMagHUmidity,
                            CMagTemperature, ChangeRate1, ChangeRate25, ChangeRate10, ChangeMagn1, ChangeMagn25,
                            ChangeMagn10, ChangeIncreaseMagn_rate1, ChangeIncreaseMagn_rate25,
                            ChangeIncreaseMagn_rate10, ChangeDecreaseMagn_rate1,
                            ChangeDecreaseMagn_rate25, ChangeDecreaseMagn_rate10, STDRATIOrate1, STDRATIOrate25,
                            STDRATIOrate10, NSTDPM2]
    print(Feautures_AA_Smok_11)

    Feautures_AA_Smok_1 = pd.concat(
        [pd.DataFrame([Feautures_AA_Smok_11], columns=Feautures_AA_Smok_1.columns), Feautures_AA_Smok_1],
        ignore_index=True)

    '''menei to 6 kai to  7'''

    Feautures_AA_Smok_1.to_csv(path2 + name2)


'''
Features_1 = df1[0:29]
print(Features_1)
'''


import itertools as it

def moving_window(x, length, step=1):
    streams = it.tee(x, length)
    return zip(*[it.islice(stream, i, None, step*length) for stream, i in zip(streams, it.count(step=step))])
x_=list(moving_window(glued_data.to_numpy(), 31))
x_=np.asarray(x_)
#print(x_)


def function(df1):
    x_prev = 0
    i = rateOfwindow
    for x in range(len(df1) // rateOfwindow):
        print(x_prev)
        Features_2 = df1[x_prev:i]
        x_prev = i
        i += rateOfwindow
        listwindow.append(Features_2)


function(glued_data)

#x_=list(moving_window(glued_data.to_numpy(), 32))



for value in listwindow:
    pds = value
    print(pds)
    func(pds)
