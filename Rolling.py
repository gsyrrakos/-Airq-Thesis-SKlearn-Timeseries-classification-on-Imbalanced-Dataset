import math
from datetime import datetime as dt
import pandas as pd
import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import math as mth
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False
rate = 12
path = "G://tinasData//normal//ola//"
name = "Sensor_AA_normal_1.csv"

df1 = pd.read_csv(path + name, parse_dates=["TimeStamp"], index_col="TimeStamp")

sc = StandardScaler()
sc.fit(df1[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']])
''''''
df1[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']] = sc.transform(
    df1[['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']])

'''
df1[['PM 2.5', 'PM 1', 'PM 10', 'Humidity', 'Temperature']] = (df1[['PM 2.5', 'PM 1', 'PM 10', 'Humidity',
                                                                    'Temperature']] - df1[
                                                                   ['PM 2.5', 'PM 1', 'PM 10', 'Humidity',
                                                                    'Temperature']].mean()) / df1[
                                                                  ['PM 2.5', 'PM 1', 'PM 10', 'Humidity',
                                                                   'Temperature']].std()

                                                            '''

Features_1 = df1


plot_cols = ['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']
plot_features = Features_1[plot_cols]
plot_features.index = Features_1.index
_ = plot_features.plot(subplots=True)
plt.show()

Features_2 = df1
print(Features_2.head(100))

Features_3 = df1
Features_4 = df1

plot_cols = ['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature']
plot_features = Features_2[plot_cols]
plot_features.index = Features_2.index
_ = plot_features.plot(subplots=True)
plt.show()

listpds = [Features_1, Features_2, Features_3]


# PM Sensors
def increaseRate(df1, val):
    rate = (abs(df1[val].max()) - abs(df1[val].iloc[0])) / abs(df1[val].iloc[0])
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
    rate = (abs(df1[val].max()) - abs(df1[val].iloc[-1]) / abs(df1[val].max()))
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


# making new csv
'''
CSV = {'Increase_RatePM1': [Increase_RatePM1], 'Increase_RatePM25': [Increase_RatePM25],
       'Increase_RatePM10': [Increase_RatePM10], 'Increase_MagnitudePM1': [Increase_MagnitudePM1],
       'Increase_MagnitudePM25': [Increase_MagnitudePM25], 'Increase_MagnitudePM10': [Increase_MagnitudePM10],
       'DeacreaseRatePM1': [DeacreaseRatePM1], 'DeacreaseRatePM25': [DeacreaseRatePM25],
       'DeacreaseRatePM10': [DeacreaseRatePM10], 'Deacrease_Magnitude1': [Deacrease_Magnitude1],
       'Deacrease_Magnitude25': [Deacrease_Magnitude25],
       'Deacrease_Magnitude10': [Deacrease_Magnitude10],
       'STDPM1': [STDPM1], 'STDPM25': [STDPM25], 'STDPM10': [STDPM10], 'STDHumidity': [STDHumidity],
       'STDTemp': [STDTemp], 'CMagHUmidity': [CMagHUmidity],
       'CMagTemperature': [CMagTemperature]

       }
Feautures_AA_Smok_1 = df = pd.DataFrame(CSV, columns=['Increase_RatePM1', 'Increase_RatePM25', 'Increase_RatePM10',
                                                       'Increase_MagnitudePM1',
                                                       'Increase_MagnitudePM25', 'Increase_MagnitudePM10',
                                                       'DeacreaseRatePM1', 'DeacreaseRatePM25',
                                                       'DeacreaseRatePM10', 'Deacrease_Magnitude1',
                                                       'Deacrease_Magnitude25', 'Deacrease_Magnitude10', 'STDPM1',
                                                       'STDPM25', 'STDPM10', 'STDHumidity', 'STDTemp', 'CMagHUmidity',
                                                       'CMagTemperature'])
print(Feautures_AA_Smok_1.head())
'''


def func(listpdf):
    for value in listpdf:
        pds = value

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
        STDHumidity = Std(pds, 'Humidity')
        STDTemp = Std(pds, 'Temperature')
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

        CMagHUmidity = ChangeMagni(pds, 'Humidity')
        CMagTemperature = ChangeMagni(pds, 'Temperature')
        NSTDPM2 = NSTD(pds)
        # Feautures_AA_Smok_1 = pd.read_csv(path + 'Feautures_AA_Smok_1.csv')
        # Feautures_AA_Smok_1 = Feautures_AA_Smok_1.drop(columns=['Unnamed: 0', ])
        CSV = {'Increase_RatePM1': [Increase_RatePM1], 'Increase_RatePM25': [Increase_RatePM25],
               'Increase_RatePM10': [Increase_RatePM10], 'Increase_MagnitudePM1': [Increase_MagnitudePM1],
               'Increase_MagnitudePM25': [Increase_MagnitudePM25], 'Increase_MagnitudePM10': [Increase_MagnitudePM10],
               'DeacreaseRatePM1': [DeacreaseRatePM1], 'DeacreaseRatePM25': [DeacreaseRatePM25],
               'DeacreaseRatePM10': [DeacreaseRatePM10], 'Deacrease_Magnitude1': [Deacrease_Magnitude1],
               'Deacrease_Magnitude25': [Deacrease_Magnitude25],
               'Deacrease_Magnitude10': [Deacrease_Magnitude10],
               'STDPM1': [STDPM1], 'STDPM25': [STDPM25], 'STDPM10': [STDPM10], 'STDHumidity': [STDHumidity],
               'STDTemp': [STDTemp], 'CMagHUmidity': [CMagHUmidity],
               'CMagTemperature': [CMagTemperature], 'ChangeRate1': [ChangeRate1], 'ChangeRate25': [ChangeRate25],
               'ChangeRate10': [ChangeRate10], 'ChangeMagn1': [ChangeMagn1], 'ChangeMagn25': [ChangeMagn25],
              'ChangeMagn10': [ChangeMagn10], 'ChangeIncreaseMagn_rate1': [ChangeIncreaseMagn_rate1],
              'ChangeIncreaseMagn_rate25': [ChangeIncreaseMagn_rate25],
              'ChangeIncreaseMagn_rate10': [ChangeIncreaseMagn_rate10], 'ChangeDecreaseMagn_rate1': [ChangeDecreaseMagn_rate1],
              'ChangeDecreaseMagn_rate25': [ChangeDecreaseMagn_rate25], 'ChangeDecreaseMagn_rate10': [ChangeDecreaseMagn_rate10],
              'STDRATIOrate1': [STDRATIOrate1], 'STDRATIOrate25': [STDRATIOrate25],
              'STDRATIOrate10': [STDRATIOrate10], 'NSTDPM2': [NSTDPM2]

        }
        Feautures_AA_Smok_1 = df = pd.DataFrame(CSV,
                                                columns=['Increase_RatePM1', 'Increase_RatePM25', 'Increase_RatePM10',
                                                         'Increase_MagnitudePM1',
                                                         'Increase_MagnitudePM25', 'Increase_MagnitudePM10',
                                                         'DeacreaseRatePM1', 'DeacreaseRatePM25',
                                                         'DeacreaseRatePM10', 'Deacrease_Magnitude1',
                                                         'Deacrease_Magnitude25', 'Deacrease_Magnitude10', 'STDPM1',
                                                         'STDPM25', 'STDPM10', 'STDHumidity', 'STDTemp', 'CMagHUmidity',
                                                         'CMagTemperature', 'ChangeRate1', 'ChangeRate25',
                                                         'ChangeRate10', 'ChangeMagn1', 'ChangeMagn25',
                                                         'ChangeMagn10', 'ChangeIncreaseMagn_rate1',
                                                         'ChangeIncreaseMagn_rate25',
                                                         'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                                                         'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10',
                                                         'STDRATIOrate1', 'STDRATIOrate25',
                                                         'STDRATIOrate10', 'NSTDPM2'])
        print(Feautures_AA_Smok_1.head())
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

        '''menei to 6 ka'''

        Feautures_AA_Smok_1.to_csv(path + 'Feautures_AA_normal_2.csv')


func(listpds)
