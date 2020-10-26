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
import sklearn
import tensorflow as tf
import pandas as pd
import glob

from imblearn.over_sampling import SMOTE

from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from Optunity import plot_learning_curve, plot_multiclass_roc

print(sklearn.__version__)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

name2 = 'Feautures_AA_Smok_1.csv'

path1 = "G://metrhseis//normal//ff//"
path2 = "G://metrhseis//smoke//ff//"
path3 = "G://metrhseis//cook//ff//"
path4 = "G://metrhseis//ypoloipa//ff//"

os.chdir(path1)
i = 0
dfNormal = pd.DataFrame()
for file in glob.glob("*.csv"):
    fil = pd.read_csv(file)
    dfNormal = pd.concat([dfNormal, fil], axis=0)

os.chdir(path3)
i = 0
dfCooking = pd.DataFrame()
for file in glob.glob("*.csv"):
    fil = pd.read_csv(file)
    dfCooking = pd.concat([dfCooking, fil], axis=0)

os.chdir(path4)
i = 0
df4 = pd.DataFrame()
for file in glob.glob("*.csv"):
    fil = pd.read_csv(file)
    df4 = pd.concat([df4, fil], axis=0)

df2 = dfNormal  # pd.read_csv(path1 + name1)
df1 = pd.read_csv(path2 + name2)
df3 = dfCooking  # pd.read_csv(path3 + name3)
df4 = df4

'''df2 = df2[(df2['NSTDPM2'] < 0.3) & (df2['Increase_RatePM25'] <= 0.3) & (df2['STDPM25'] <= 5)]  # normal
df1 = df1[(df1['NSTDPM2'] > 0.3) & (df1['Increase_RatePM25'] >= 2.5)]
df3 = df3[(df3['NSTDPM2'] > 0.3) & (df3['Increase_RatePM25'] <= 2.5) & (df3['Increase_RatePM25'] >= 0.3)]

df4cook = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] <= 2.5)]

df4smok = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] >= 2.5)]
 random_state=1, max_depth=15, n_estimators=800, min_samples_split=10,
                                   min_samples_leaf=1'''

df1 = df1[(df1['NSTDPM2'] > 0.8) ]
df3 = df3[(df3['NSTDPM2'] < 1)  ]

df4cook = df4[(df4['NSTDPM2'] < 1)  ]

df4smok = df4[(df4['NSTDPM2'] > 0.8) ]



print(df1.shape)
print(df3.shape)
print(df2.shape)

df1['Target'] = 'Smoking'
df2['Target'] = 'normal'
df3['Target'] = 'Cooking'
df4smok['Target'] = 'Smoking'

df4cook['Target'] = 'Cooking'
#

glued_data = pd.DataFrame()
glued_data = pd.concat([glued_data, df1], axis=0)
# glued_data = pd.concat([glued_data, df2], axis=0)
glued_data = pd.concat([glued_data, df3], axis=0)
glued_data = pd.concat([glued_data, df4smok], axis=0)
# glued_data = pd.concat([glued_data, df4norm], axis=0)
glued_data = pd.concat([glued_data, df4cook], axis=0)
glued_data = glued_data[~glued_data.isin([np.nan, np.inf, -np.inf]).any(1)]


le = preprocessing.LabelEncoder()
le.fit(glued_data[['Target']])
glued_data['Target'] = le.transform(glued_data[['Target']])
#print(glued_data)
'''
glued_data['Target']=le.inverse_transform(glued_data[['Target']])
print(glued_data)
'''
glued_data = glued_data.drop('NSTDPM2', axis=1)
glued_data = glued_data.drop('Unnamed: 0', axis=1)
'''
glued_data = glued_data.drop(['STDHumidity', 'STDTemp', 'CMagHUmidity',
                              'CMagTemperature', 'ChangeRate1', 'ChangeRate25', 'ChangeRate10', 'ChangeMagn1',
                              'ChangeMagn25',
                              'ChangeMagn10', 'ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
                              'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                              'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10', 'STDRATIOrate1',
                              'STDRATIOrate25',
                              'STDRATIOrate10'], axis=1)
                              
                              '''
print(glued_data)
X = glued_data.drop('Target', axis=1)
y = glued_data['Target']
# print(X)
# print(y)
rand = ExtraTreesClassifier(n_estimators=50, random_state=10)  # RandomForestClassifier(n_estimators=100)
clf = rand.fit(X, y)
import joblib

joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbRandFFimpo1.pkl')
sel = SelectFromModel(clf, prefit=True)
X = sel.transform(X)

training_features, test_features, training_target, test_target, = train_test_split(X, y, test_size=.1, random_state=12)
strategy = {0: 800, 1: 800}
sm = SMOTE(sampling_strategy='all', k_neighbors=9, random_state=10)

x_res, y_res = sm.fit_resample(training_features, training_target)
# print(np.bincount(y_res))
# print(training_target.value_counts())


X = x_res
y = y_res
# function(X, 'sc')


strategy = {0: 200, 1: 200}
sm1 = SMOTE(sampling_strategy='all', k_neighbors=2, random_state=10)

xt, yt = sm1.fit_resample(test_features, test_target)
test_features = xt
test_target = yt

# function(X, 'sc')
lsvc = LinearSVC(C=0.05, penalty="l1", dual=False)
rand = RandomForestClassifier(n_estimators=100)
sel = SelectFromModel(rand)
# X=sel.fit_transform(X, y)
# print(X.shape)

# ALgorithm KNN
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


X_train = X
y_train = y
X_test = test_features
y_test = test_target

lsvc = LinearSVC(C=0.05, penalty="l1", dual=False)
rand = RandomForestClassifier(n_estimators=100)
sel = SelectFromModel(rand)
# X=sel.fit_transform(X, y)
print(X.shape)

# ALgorithm Rndomw
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

forest = RandomForestClassifier(random_state=1)
modelF = forest.fit(X_train, y_train)

# tuning

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]
'''
hyperF = dict(n_estimators=n_estimators, max_depth=max_depth,
              min_samples_split=min_samples_split,
              min_samples_leaf=min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv=3, verbose=1,
                     n_jobs=-1)
bestF = gridF.fit(X_train, y_train)
print(bestF.best_params_)
'''
# {random_state=1, max_depth=15, n_estimators=800, min_samples_split=2,min_samples_leaf=1)}
#                                    {'max_depth': 8, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 800}
'''forestOpt = RandomForestClassifier(random_state=1, max_depth=15, n_estimators=800, min_samples_split=2,
                                   min_samples_leaf=1)
                                   
                                   random_state=1, max_depth=25, n_estimators=100, min_samples_split=5,
                                   min_samples_leaf=2
                                   random_state=1, max_depth=15, n_estimators=800, min_samples_split=10,
                                   min_samples_leaf=1
                                   RandomForestClassifier(random_state=1, max_depth=5, n_estimators=1200, min_samples_split=2,
                                   min_samples_leaf=5)
                                   '''
forestOpt = RandomForestClassifier( random_state=1, max_depth=25, n_estimators=100, min_samples_split=5,
                                   min_samples_leaf=2)

modelOpt = forestOpt.fit(X_train, y_train)
pred = modelOpt.predict(test_features)
print("Accuracy:{}".format(accuracy_score(test_target, pred)))


# test_features, test_target
def cm2df(cm, labels):
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata = {}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label] = cm[i, j]
        df = df.append(pd.DataFrame.from_dict({row_label: rowdata}, orient='index'))
    return df[labels]


cm = confusion_matrix(test_target, pred)
df = cm2df(cm, le.classes_)
print(df)
print(np.unique(pred))
# now will try the same with joblib


joblib.dump(modelOpt, 'C://Users//giorgos//Desktop//AIR_quallityjobilbRandwithff1.pkl')


# load and test this

# whatever = joblib.load('AIR_quallityjobilb.pkl')

# Ypredict = whatever.predict(X_test)
# print(Ypredict)
# print (joblib.__version__)


def get_integer_mapping(le):
    '''
    Return a dict mapping labels to their integer values
    from an SKlearn LabelEncoder
    le = a fitted SKlearn LabelEncoder
    '''
    res = {}
    for cl in le.classes_:
        res.update({cl: le.transform([cl])[0]})

    return res


integerMapping = get_integer_mapping(le)
print(integerMapping['Smoking'])  # Returns 0
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = r"Learning Curves (Random forest, Neibors=7)"
plot_learning_curve(modelOpt, title, X_train, y_train, X_test, y_test, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=3, n_jobs=-1)

plot_multiclass_roc(modelOpt, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()
