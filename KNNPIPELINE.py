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
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score, \
    plot_precision_recall_curve, average_precision_score, precision_recall_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from Optunity import  plot_multiclass_roc

print(sklearn.__version__)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

name2 = 'Feautures_AA_Smok_1.csv'

path1 = "G://metrhseis//normal//ff//"
path2 = "G://metrhseis//smoke//ff//"
path3 = "G://metrhseis//cook//ff//"
path4 = "G://metrhseis//ypoloipa//ff//"
name5='Feautures_AA_smok1_1.csv'
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
df5 = pd.read_csv(path2 + name5)

df1 =     df1[(df1['STDPM25'] >= 35) ]
df5 =     df5[(df5['STDPM25'] >= 35) ]
df4smok = df4[(df4['STDPM25'] >= 35)]

df3 = df3[(df3['Increase_RatePM25'] < 17) ]

df4cook = df4[(df4['Increase_RatePM25'] < 17) ]


print(df1.shape)
print(df3.shape)

df1['Target'] = 'Smoking'

df3['Target'] = 'Cooking'
df4smok['Target'] = 'Smoking'
df5['Target'] = 'Smoking'
df4cook['Target'] = 'Cooking'
#

glued_data = pd.DataFrame()
glued_data = pd.concat([glued_data, df1], axis=0)
glued_data = pd.concat([glued_data, df5], axis=0)
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
glued_data = glued_data.drop(['STDHumidity', 'STDTemp', 'CMagHUmidity',
                              'CMagTemperature', 'ChangeRate1', 'ChangeRate25', 'ChangeRate10', 'ChangeMagn1',
                              'ChangeMagn25',
                              'ChangeMagn10', 'ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
                              'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                              'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10', 'STDRATIOrate1',
                              'STDRATIOrate25',
                              'STDRATIOrate10',
                              'DeacreaseRatePM1', 'DeacreaseRatePM25',
                              'DeacreaseRatePM10', 'Deacrease_Magnitude1',
                              'Deacrease_Magnitude25', 'Deacrease_Magnitude10',
                              'Increase_MagnitudePM1',
                              'Increase_MagnitudePM25', 'Increase_MagnitudePM10'], axis=1)


# print(glued_data)
X = glued_data.drop('Target', axis=1)
y = glued_data['Target']
# print(X)
# print(y)


import joblib

rand = ExtraTreesClassifier(n_estimators=50, random_state=42)  # RandomForestClassifier(n_estimators=100)
clf = rand.fit(X, y)
joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNNFFimpo1pipeline.pkl')
sel = SelectFromModel(clf, prefit=True)
X = sel.transform(X)

# X=sel.fit_transform(X, y)
print(X.shape)

# tuning
print(X.shape)


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


# load and test this

# whatever = joblib.load('AIR_quallityjobilb.pkl')

# Ypredict = whatever.predict(X_test)
# print(Ypredict)
# print (joblib.__version__)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler
oversample = BorderlineSMOTE(random_state=10)
model = Pipeline([#('scaler',MinMaxScaler(feature_range=(0,1))),
    ('resample', oversample),
    ('model', KNeighborsClassifier())
])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(
    estimator=model, param_grid={'model__n_neighbors': k_range},scoring='balanced_accuracy', cv=kf, refit=True
)
grid_search.fit(X_train, y_train)

# Adding below in as could be helpful to know how to get fitted scaler if used
best = grid_search.best_estimator_
#print(best['scaler'])

#scal = best['scaler']
#X_test=scal.transform(X_test)
pred = best.predict(X_test)

from sklearn.metrics import accuracy_score

print("Accuracy:{}".format(accuracy_score(y_test, pred)))
cm = confusion_matrix(y_test, pred)
df = cm2df(cm, le.classes_)
print(df)
# grid = GridSearchCV(knn, param_grid, cv=10, scoring='accuracy',verbose=2)
# grid.fit(X_train, y_train)
# examine the best model

# print(grid.best_params_)

# joblib.dump(scal, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNNscaler.pkl')
joblib.dump(best, 'C://Users//giorgos//Desktop//AIR_quallityjobilbKNN4pipeline.pkl')

score = balanced_accuracy_score(y_test, pred)
print(f"Balanced accuracy score of a dummy classifier: {score:.3f}")

print('The geometric mean is {}'.format(geometric_mean_score(
    y_test,
    pred)))



plot_multiclass_roc(best, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()


print(classification_report(y_test, pred))

probs = best.predict_proba(X_test)
probs = probs[:, 1]

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