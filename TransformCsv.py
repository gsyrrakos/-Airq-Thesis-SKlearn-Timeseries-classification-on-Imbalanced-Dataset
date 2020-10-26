import pandas as pd
import glob
import os
import os
import glob
import pandas as pd
import re
from datetime import datetime as dt

import pandas as pd
import glob

path = "G://MyData//"
name = "Sensor_AA_Smoking_"

import glob, os

os.chdir(path)
i = 0
for file in glob.glob("*.csv"):
    df1 = pd.read_csv(file)

    df1 = df1.rename(columns={'false.2': 'TimeStamp'})
    df1['TimeStamp'] = pd.to_datetime(df1["TimeStamp"], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

    # creating a blank series
    Type_new1 = pd.Series([])
    Type_new2 = pd.Series([])
    Type_new3 = pd.Series([])
    Type_new4 = pd.Series([])
    Type_new5 = pd.Series([])

    index = []

    a = 0
    a1 = 0
    a2 = 0
    a3 = 0
    a4 = 0
    a5 = 0

    # running a for loop and asigning some values to series
    for ind in range(len(df1)):
        if df1['true.3'][ind] == 'PM 10':
            if len(index) != 0:
                a5 += 1
                Type_new1[index[a5 - 1]] = df1['false.3'][ind]
                index.append(ind)
            else:
                Type_new1[ind] = df1['false.3'][ind]
                index.append(ind)


        elif df1['true.3'][ind] == 'Temperature':
            if len(index) != 0:
                a += 1
                Type_new2[index[a - 1]] = df1['false.3'][ind]
                index.append(ind)
            else:
                Type_new1[ind] = df1['false.3'][ind]
                index.append(ind)



        elif df1['true.3'][ind] == 'PM 2.5':
            if len(index) != 0:
                a1 += 1
                Type_new3[index[a1 - 1]] = df1['false.3'][ind]
                index.append(ind)
            else:
                Type_new3[ind] = df1['false.3'][ind]
                index.append(ind)



        elif df1['true.3'][ind] == 'Humidity':
            if len(index) != 0:
                a2 += 1
                Type_new4[index[a2 - 1]] = df1['false.3'][ind]
                index.append(ind)
            else:
                Type_new4[ind] = df1['false.3'][ind]
                index.append(ind)




        elif df1['true.3'][ind] == 'PM 1':
            if len(index) != 0:
                a3 += 1
                Type_new5[index[a3 - 1]] = df1['false.3'][ind]
                index.append(ind)
            else:
                Type_new5[ind] = df1['false.3'][ind]
                index.append(ind)


        else:
            print('d')

        # inserting new column with values of list made above
    df1.insert(10, "PM 10", Type_new1)
    df1.insert(11, "Temperature", Type_new2)
    df1.insert(12, "PM 2.5", Type_new3)
    df1.insert(13, "Humidity", Type_new4)
    df1.insert(14, "PM 1", Type_new5)

    # list output
    df1.head()
    i = i + 1

    cols_to_keep = ['PM 1', 'PM 2.5', 'PM 10', 'Humidity', 'Temperature', 'TimeStamp']
    df1.dropna(how='all')
    print(df1[cols_to_keep].head())

    # https://stackabuse.com/tensorflow-2-0-solving-classification-and-regression-problems/

    df1[cols_to_keep] = df1[cols_to_keep].dropna()
    namee = name + str(i) + ".csv"
    df1[cols_to_keep].to_csv(path + namee)
