
import json

import pandas as pd
import glob
import os
import os
import glob
import pandas as pd
import re
from datetime import datetime as dt
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import pandas as pd
import glob
import math
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import MinMaxScaler
#----------------------------------------------------------------------------------------------------------------------------
# PM Sensors

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
    #print(abs(df1[val].iloc[-1]))
    #print(abs(df1[val].max()))
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


#------------------------------------------------------------------------------------------------------------
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
                            STDRATIOrate10]
    print(Feautures_AA_Smok_11)

    Feautures_AA = pd.DataFrame([Feautures_AA_Smok_11],
                                columns=['Increase_RatePM1', 'Increase_RatePM25', 'Increase_RatePM10',
                                         'Increase_MagnitudePM1',
                                         'Increase_MagnitudePM25', 'Increase_MagnitudePM10',
                                         'DeacreaseRatePM1', 'DeacreaseRatePM25',
                                         'DeacreaseRatePM10', 'Deacrease_Magnitude1',
                                         'Deacrease_Magnitude25', 'Deacrease_Magnitude10', 'STDPM1',
                                         'STDPM25', 'STDPM10','STDHumidity', 'STDTemp', 'CMagHUmidity',
                                         'CMagTemperature', 'ChangeRate1', 'ChangeRate25', 'ChangeRate10',
                                         'ChangeMagn1',
                                         'ChangeMagn25',
                                         'ChangeMagn10', 'ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
                                         'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                                         'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10', 'STDRATIOrate1',
                                         'STDRATIOrate25',
                                         'STDRATIOrate10'])
    return Feautures_AA


def funcsvm(listpdf):
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



    Feautures_AA_Smok_11 = [Increase_RatePM1, Increase_RatePM25, Increase_RatePM10,
                            Increase_MagnitudePM1,
                            Increase_MagnitudePM25, Increase_MagnitudePM10,
                             Deacrease_Magnitude1,
                            Deacrease_Magnitude25, Deacrease_Magnitude10, STDPM1,
                            STDPM25, STDPM10]
    print(Feautures_AA_Smok_11)

    Feautures_AA = pd.DataFrame([Feautures_AA_Smok_11],
                                columns=['Increase_RatePM1', 'Increase_RatePM25', 'Increase_RatePM10',
                                         'Increase_MagnitudePM1',
                                         'Increase_MagnitudePM25', 'Increase_MagnitudePM10',
                                         'Deacrease_Magnitude1',
                                         'Deacrease_Magnitude25', 'Deacrease_Magnitude10', 'STDPM1',
                                         'STDPM25', 'STDPM10'])
    return Feautures_AA



#-----------------------------------------------------------------------------------------------------------

def func1(listpdf):
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
    STDPM1 = Std(pds, 'PM 1')
    STDPM25 = Std(pds, 'PM 2.5')
    STDPM10 = Std(pds, 'PM 10')
    NSTDPM2 = NSTD(pds)


    Feautures_AA_Smok_11 = [Increase_RatePM1, Increase_RatePM25, Increase_RatePM10, STDPM1,
                            STDPM25, STDPM10

                            ]
    print(Feautures_AA_Smok_11)

    Feautures_AA = pd.DataFrame([Feautures_AA_Smok_11],
                                columns=[ 'Increase_RatePM1','Increase_RatePM25', 'Increase_RatePM10', 'STDPM1',
                                         'STDPM25', 'STDPM10'
                                          ])
    return Feautures_AA


#---------------------------------------------------------------------------------------------------------------------------------
def getDtat(jso):
    # edw einai o pinkaass ths m

    d = {}
    for x in range(0, 6):
        pairs11 = zip(jso[6], jso[x])
        json_val = ('"{}": {} '.format(label, value) for label, value in pairs11)
        value = "{" + ", ".join(json_val) + "}"
        d["value" + str(x)] = value

    my_array = [[d["value0"], d["value1"], d["value2"], d["value3"], d["value4"], d["value5"]],
                ["Unnamed: 0", "PM 1", "PM 2.5", "PM 10", "Humidity", "Temperature"]]

    pairs = zip(my_array[1], my_array[0])
    json_values = ('"{}": {}'.format(label, value) for label, value in pairs)
    my_string = "{" + ", ".join(json_values) + "}"

    my_string = my_string.replace("'", '"')

    return my_string
