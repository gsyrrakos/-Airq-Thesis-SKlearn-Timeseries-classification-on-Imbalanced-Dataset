import sys
import os
import shutil
import time
import traceback

import joblib
import sklearn
from flask import Flask, request, jsonify
import pandas as pd

import requests, json, re
from pandas.io.json import json_normalize




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



url = 'https://gsyrrakos.pythonanywhere.com/preproc'  # to link gia to preorpocces
#url = 'https://gsyrrakos.pythonanywhere.com/preprocknn'
url = 'https://gsyrrakos.pythonanywhere.com/preprocsvm'
mystr = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 25, 96, 89, 52, 9, 85, 89, 2, 82, 1, 2, 3, 4, 5, 6, 7, 8],
 [13, 11, 11, 11, 10, 11, 13, 11, 10, 11, 9, 10, 11, 11, 11, 12, 13, 14, 13, 12, 11, 12, 12, 14, 15, 14, 13, 150, 120],
 [15, 12, 12, 13, 12, 12, 15, 14, 12, 13, 11, 13, 13, 13, 13, 14, 16, 17, 15, 14, 13, 13, 15, 16, 18, 16, 15, 220, 170],
 [17, 12, 13, 15, 14, 12, 15, 15, 14, 14, 13, 13, 14, 15, 15, 14, 16, 18, 16, 15, 13, 14, 15, 17, 19, 17, 17, 190, 195],
 [51.6, 51.6, 51.6, 51.6, 51.7, 51.7, 51.8, 51.4, 51.4, 51.2, 51.1, 51.1, 51.3, 51.4, 51.5, 51.4, 51.5, 51.5, 51.5,
  51.4, 51.4, 51.5, 51.6, 51.7, 51.8, 51.8, 51.6, 51.6, 51.8],
 [25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.1, 25.1, 25.1, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2,
  25.2, 25.2, 25.2, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3],
 ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
  "21", "22", "23", "24", "25", "26", "27", "28", "29"]]

'''
[[1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 14, 15, 25, 96, 89, 52, 9, 85, 89, 2, 82, 1, 2, 3, 4, 5, 6, 7, 8],
 [13, 11, 11, 11, 10, 11, 13, 11, 10, 11, 9, 10, 11, 11, 11, 12, 13, 14, 13, 12, 11, 12, 12, 14, 15, 14, 13, 11, 13],
 [15, 12, 12, 13, 12, 12, 15, 14, 12, 13, 11, 13, 13, 13, 13, 14, 16, 17, 15, 14, 13, 13, 15, 16, 18, 16, 15, 14, 16],
 [17, 12, 13, 15, 14, 12, 15, 15, 14, 14, 13, 13, 14, 15, 15, 14, 16, 18, 16, 15, 13, 14, 15, 17, 19, 17, 17, 16, 16],
 [51.6, 51.6, 51.6, 51.6, 51.7, 51.7, 51.8, 51.4, 51.4, 51.2, 51.1, 51.1, 51.3, 51.4, 51.5, 51.4, 51.5, 51.5, 51.5,
  51.4, 51.4, 51.5, 51.6, 51.7, 51.8, 51.8, 51.6, 51.6, 51.8],
 [25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2, 25.1, 25.1, 25.1, 25.2, 25.2, 25.2, 25.2, 25.2, 25.2,
  25.2, 25.2, 25.2, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3, 25.3],
 ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
  "21", "22", "23", "24", "25", "26", "27", "28", "29"]]'''

jsok = json.dumps(mystr)
print('edw')
print(jsok)
data = json.loads(jsok)
jso = data

headers = {'Content-type': 'application/json'}
r = requests.post(url, json=jsok)  # to stelnw

print(r.json())  # pairnv pisw tin apanthsh apo to preprocces




