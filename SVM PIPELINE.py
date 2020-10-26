import joblib
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

from imblearn.combine import SMOTEENN
from imblearn.metrics import geometric_mean_score
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from matplotlib import gridspec
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler, RobustScaler, FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, \
    average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from Optunity import plot_learning_curve, plot_multiclass_roc, plot_learning_curve1, ffImportance, SeabornFig2Grid
from Optunity import SeabornFig2Grid as sfg

print(sklearn.__version__)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

name2 = 'Feautures_AA_Smok_1.csv'

path1 = "G://metrhseis//normal//ff//"
path2 = "G://metrhseis//smoke//ff//"
path3 = "G://metrhseis//cook//ff//"
path4 = "G://metrhseis//ypoloipa//ff//"
name5 = 'Feautures_AA_smok1_1.csv'
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

''''''
df5 = pd.read_csv(path2 + name5)

'''
df1 =     df1[(df1['Increase_RatePM25'] >= 17) & (df1['Increase_RatePM1'] >= 13) & (df1['Increase_RatePM10'] >= 16)]
df5 =     df5[(df5['Increase_RatePM25'] >= 17) & (df5['Increase_RatePM1'] >= 13) & (df5['Increase_RatePM10'] >= 16)]
df4smok = df4[(df4['Increase_RatePM25'] >= 17) & (df4['Increase_RatePM1'] >= 13) & (df4['Increase_RatePM10'] >= 16)]

df3 = df3[(df3['Increase_RatePM25'] < 13) & (df3['Increase_RatePM1'] < 13) & (df3['Increase_RatePM10'] < 13)]

df4cook = df4[(df4['Increase_RatePM25'] < 13) & (df4['Increase_RatePM1'] < 13) & (df4['Increase_RatePM10'] < 13)]


df1 =     df1[(df1['Increase_RatePM25'] >= 17) & (df1['Increase_RatePM1'] >= 13) & (df1['Increase_RatePM10'] >= 16)]
df5 =     df5[(df5['Increase_RatePM25'] >= 17) & (df5['Increase_RatePM1'] >= 13) & (df5['Increase_RatePM10'] >= 16)]
df4smok = df4[(df4['Increase_RatePM25'] >= 17) & (df4['Increase_RatePM1'] >= 13) & (df4['Increase_RatePM10'] >= 16)]

df3 = df3[(df3['Increase_RatePM25'] < 13) & (df3['Increase_RatePM1'] < 13) & (df3['Increase_RatePM10'] < 13)]

df4cook = df4[(df4['Increase_RatePM25'] < 13) & (df4['Increase_RatePM1'] < 13) & (df4['Increase_RatePM10'] < 13)]

'''
'''
df1 =     df1[(df1['Increase_RatePM25'] >= 17) & (df1['Increase_RatePM1'] >= 13) & (df1['Increase_RatePM10'] >= 16)]
df5 =     df5[(df5['Increase_RatePM25'] >= 17) & (df5['Increase_RatePM1'] >= 13) & (df5['Increase_RatePM10'] >= 16)]
df4smok = df4[(df4['Increase_RatePM25'] >= 17) & (df4['Increase_RatePM1'] >= 13) & (df4['Increase_RatePM10'] >= 16)]

df3 = df3[(df3['Increase_RatePM25'] < 13) & (df3['Increase_RatePM1'] < 13) & (df3['Increase_RatePM10'] < 13)]

df4cook = df4[(df4['Increase_RatePM25'] < 13) & (df4['Increase_RatePM1'] < 13) & (df4['Increase_RatePM10'] < 13)]
'''
df1 = df1[(df1['STDPM25'] >= 45)]
df5 = df5[(df5['STDPM25'] >= 45)]
df4smok = df4[(df4['STDPM25'] >= 45)]

df3 = df3[(df3['Increase_RatePM25'] < 17)]

df4cook = df4[(df4['Increase_RatePM25'] < 17)]

print(df1.shape)
print(df3.shape)
print(df5.shape)
print(df4smok.shape)

df1['Target'] = 'Smoking'
df5['Target'] = 'Smoking'
df4smok['Target'] = 'Smoking'

df3['Target'] = 'Cooking'

df4cook['Target'] = 'Cooking'
#

glued_data = pd.DataFrame()
glued_data = pd.concat([glued_data, df1], axis=0)

glued_data = pd.concat([glued_data, df3], axis=0)
glued_data = pd.concat([glued_data, df4smok], axis=0)
glued_data = pd.concat([glued_data, df5], axis=0)
glued_data = pd.concat([glued_data, df4cook], axis=0)
glued_data = glued_data[~glued_data.isin([np.nan, np.inf, -np.inf]).any(1)]

g0 = sns.lmplot('NSTDPM2', 'STDPM25', glued_data, hue='Target', fit_reg=False)
g0 = sns.lmplot('NSTDPM2', 'Increase_RatePM25', glued_data, hue='Target', fit_reg=False)

arr = ['ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
       'STDPM1',
       'STDPM25', 'STDPM10',
       'DeacreaseRatePM1', 'DeacreaseRatePM25',
       'DeacreaseRatePM10', 'Deacrease_Magnitude1',
       'Deacrease_Magnitude25', 'Deacrease_Magnitude10',
       'Increase_MagnitudePM1',
       'Increase_MagnitudePM25', 'Increase_MagnitudePM10']


def plot():
    array = []
    i = 0
    for img in arr:
        g0 = sns.lmplot('NSTDPM2', img, glued_data, hue='Target', fit_reg=False)
        array.append(g0)

    fig = plt.figure(figsize=(13, 8))
    gs = gridspec.GridSpec(3, 5)
    for img in array:
        mg0 = SeabornFig2Grid(img, fig, gs[i])
        i = i + 1

    gs.tight_layout(fig)
    # gs.update(top=0.7)

    plt.show()

plot()

le = preprocessing.LabelEncoder()
le.fit(glued_data[['Target']])
glued_data['Target'] = le.transform(glued_data[['Target']])
print(glued_data)
'''
glued_data['Target']=le.inverse_transform(glued_data[['Target']])
print(glued_data)
'''
glued_data = glued_data.drop('NSTDPM2', axis=1)
glued_data = glued_data.drop('Unnamed: 0', axis=1)

glued_data = glued_data.drop(['STDHumidity', 'STDTemp', 'CMagHUmidity',
                              'CMagTemperature', 'ChangeRate1', 'ChangeRate25', 'ChangeRate10', 'ChangeMagn1',
                              'ChangeMagn25',
                              'ChangeMagn10', 'ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
                              'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                              'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10', 'STDRATIOrate1',
                              'STDRATIOrate25',
                              'STDRATIOrate10',
                              'DeacreaseRatePM1', 'DeacreaseRatePM25',
                              'DeacreaseRatePM10'], axis=1)
''', 'Deacrease_Magnitude1',
                              'Deacrease_Magnitude25', 'Deacrease_Magnitude10',
                              'Increase_MagnitudePM1',
                              'Increase_MagnitudePM25', 'Increase_MagnitudePM10'''

print(glued_data)
X = glued_data.drop('Target', axis=1)
y = glued_data['Target']
# print(X)
# print(y)


rand = ExtraTreesClassifier(n_estimators=50, random_state=10)  # RandomForestClassifier(n_estimators=100)
clf = rand.fit(X, y)

ffImportance(clf, X)

import joblib

joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbSVMFFimpo12.pkl')
sel = SelectFromModel(clf, prefit=True)
X = sel.transform(X)

print(X.shape)
# ALgorithm

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
scal = StandardScaler()
scal.fit(X_train)
joblib.dump(scal, 'C://Users//giorgos//Desktop//AIR_quallityjobilb_SVMFFpipelinescaler.pkl')
X_train = scal.transform(X_train)


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


from sklearn.metrics import accuracy_score

# tuning hyperparamaters for SVM
# X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.2, random_state=42)
param_grid = {'C': [0.1, 10, 25, 50, 100, 1000],
              'kernel': ['rbf', 'sigmoid']}

parameters = {
    "model__C": [0.1, 10, 25, 50, 100, 1000],
    "model__kernel": ['rbf', 'sigmoid'],
    "model__gamma": [1, 0.1, 0.01, 0.001, 0.0001]

}

sme = SMOTE(k_neighbors=5, random_state=42)

oversample = BorderlineSMOTE(random_state=10, n_jobs=-1)
oversample = SVMSMOTE(random_state=10)
over = ADASYN(random_state=10)

model = Pipeline([  # ('scaler', StandardScaler(with_std=False)),
    ('resample', oversample),
    ('model', SVC(cache_size=1000, probability=True))
])

kf = StratifiedKFold(n_splits=3, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=parameters, scoring='balanced_accuracy', refit=True, cv=kf, verbose=2,
                    n_jobs=-1)
grid.fit(X_train, y_train)
# scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
# model_to_set = OneVsRestClassifier(SVC())

print(grid.best_params_)
best = grid.best_estimator_

# scal = best['scaler']
X_test = scal.transform(X_test)
pred = best.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:{}".format(accuracy_score(y_test, pred)))
cm = confusion_matrix(y_test, pred)
df = cm2df(cm, le.classes_)
print(df)

# {'model__C': 1000, 'model__gamma': 1, 'model__kernel': 'rbf'}
# {'model__C': 1000, 'model__gamma': 0.1, 'model__kernel': 'rbf'}


score = balanced_accuracy_score(y_test, pred)
print(f"Balanced accuracy score of a dummy classifier: {score:.3f}")

print('The geometric mean is {}'.format(geometric_mean_score(
    y_test,
    pred)))

X = scal.transform(X)
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = r"Learning Curves (Knn, Neibors=7)"
plot_learning_curve1(best, 'title', X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                     cv=5, n_jobs=-1)
plot_multiclass_roc(best, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()
joblib.dump(best, 'C://Users//giorgos//Desktop//AIR_quallityjobilb_SVMFFpipeline.pkl')
print(classification_report(y_test, pred))

probs = best.decision_function(X_test)

average_precision = average_precision_score(y_test, probs)

print('Average precision-recall score: {0:0.2f}'.format(
    average_precision))

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import matplotlib.pyplot as plt

disp = plot_precision_recall_curve(best, X_test, y_test)
disp.ax_.set_title('2-class Precision-Recall curve: '
                   'AP={0:0.2f}'.format(average_precision))
plt.show()
