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
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
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

'''
                                   min_samples_leaf=1'''


df1 = df1[(df1['NSTDPM2'] > 0.3) & (df1['Increase_RatePM25'] >= 2.5)]
df3 = df3[(df3['NSTDPM2'] > 0.3) & (df3['Increase_RatePM25'] <= 2.5) & (df3['Increase_RatePM25'] >= 0.3)]

df4cook = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] <= 2.5)]

df4smok = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] >= 2.5)]


print(df1.shape)
print(df3.shape)

df1['Target'] = 'Smoking'

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
# print(glued_data)
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
# print(glued_data)
X = glued_data.drop('Target', axis=1)
y = glued_data['Target']
# print(X)
# print(y)

#scaler = MinMaxScaler(feature_range=(0, 10))
scaler = Normalizer(norm='l2')
#scaler = Normalizer(norm='l1')

#scaler = StandardScaler(with_mean=False)
#scaler = MaxAbsScaler()


import joblib

rand = ExtraTreesClassifier(n_estimators=50, random_state=10)  # RandomForestClassifier(n_estimators=100)
clf = rand.fit(X, y)
joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNNFFimpo1.pkl')
sel = SelectFromModel(clf, prefit=True)
X = sel.transform(X)

# X=sel.fit_transform(X, y)
print(X.shape)

training_features, test_features, training_target, test_target, = train_test_split(X, y, test_size=.1,
                                                                                   random_state=12)
strategy = {0: 500, 1: 500}
sm = SMOTE(sampling_strategy='all', k_neighbors=3, random_state=10)

x_res, y_res = sm.fit_resample(training_features, training_target)
print(np.bincount(y_res))
# print(training_target.value_counts())


X = x_res
y = y_res

strategy = {0: 1000, 1: 1000}
sm1 = SMOTE(sampling_strategy='all', k_neighbors=1, random_state=10)

xt, yt = sm1.fit_resample(test_features, test_target)
test_features = xt
test_target = yt

# ALgorithm KNN
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler.fit(X)

rescaledX = scaler.transform(X)
joblib.dump(scaler, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNNscaler.pkl')



#X = rescaledX
#test_features=scaler.transform(test_features)

X_train = X
y_train = y
X_test = test_features
y_test = test_target



# tuning
print(X.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

# Creating odd list K for KNN
neighbors = list(range(1, 50, 2))
# empty list that will hold cv scores
cv_scores = []
# perform 10-fold cross-validation
for K in neighbors:
    knn = KNeighborsClassifier(n_neighbors=K)
    scores = cross_val_score(knn, X_train, y_train, cv=3, scoring=
    'accuracy')
    cv_scores.append(scores.mean())

# Changing to mis classification error
mse = [1 - x for x in cv_scores]
# determing best k
optimal_k = neighbors[mse.index(min(mse))]
print("The optimal no. of neighbors is {}".format(optimal_k))

import matplotlib.pyplot as plt


def plot_accuracy(knn_list_scores):
    pd.DataFrame({"K": [i for i in range(1, 50, 2)], "Accuracy": knn_list_scores}).set_index("K").plot.bar(
        figsize=(9, 6), ylim=(0.78, 0.99), rot=0)
    plt.show()


plot_accuracy(cv_scores)

# blepw oti to kalutero K einai =3 me 97% acc
from sklearn.neighbors import KNeighborsClassifier

# instantiate learning model with k=3
knn = KNeighborsClassifier(n_neighbors=12)
# fitting model
knn.fit(X_train, y_train)
# predict
pred = knn.predict(X_test)
print(np.unique(pred))

from sklearn.metrics import accuracy_score

print("Accuracy:{}".format(accuracy_score(y_test, pred)))


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


cm = confusion_matrix(y_test, pred)
df = cm2df(cm, le.classes_)
print(df)
# now will try the same with joblib


joblib.dump(knn, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNN41.pkl')

# --------------------------------------------------------------------------------------------------------

# define the parameter values that should be searched
k_range = list(range(1, 31))

# Another parameter besides k that we might vary is the weights parameters
# default options --> uniform (all points in the neighborhood are weighted equally)
# another option --> distance (weights closer neighbors more heavily than further neighbors)

# we create a list
weight_options = ['uniform', 'distance']
# create a parameter grid: map the parameter names to the values that should be searched
# dictionary = dict(key=values, key=values)
param_grid = dict(n_neighbors=k_range, weights=weight_options)
print(param_grid)



#grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',verbose=2)
#grid.fit(X_train, y_train)
# examine the best model
#print(grid.best_score_)
#print(grid.best_params_)


knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train,y_train)
pred = knn.predict(X_test)
print(np.unique(pred))

from sklearn.metrics import accuracy_score

print("Accuracy:{}".format(accuracy_score(y_test, pred)))


fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = r"Learning Curves (Knn, Neibors=7)"
plot_learning_curve(knn, title, X_train, y_train, test_features, test_target, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=3, n_jobs=-1)

plot_multiclass_roc(knn, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()
