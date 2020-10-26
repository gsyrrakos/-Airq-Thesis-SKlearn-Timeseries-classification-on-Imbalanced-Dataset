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
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE
from imblearn.pipeline import Pipeline
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from Optunity import plot_learning_curve, plot_multiclass_roc, plot_learning_curve1

print(sklearn.__version__)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

name2 = 'Feautures_AA_Smok_1.csv'
name5='Feautures_AA_smok1_1.csv'

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
df5 = pd.read_csv(path2 + name5)
df3 = dfCooking  # pd.read_csv(path3 + name3)
df4 = df4
'''
df1 = df1[(df1['NSTDPM2'] > 1)]
df5 = df5[(df5['NSTDPM2'] > 1)]
df4smok = df4[(df4['NSTDPM2'] > 1)]
df3 = df3[(df3['NSTDPM2'] < 1.20)]

df4cook = df4[(df4['NSTDPM2'] < 1.20)]




df1 =     df1[(df1['Increase_RatePM25'] >= 17) ]
df5 =     df5[(df5['Increase_RatePM25'] >= 17) ]
df4smok = df4[(df4['Increase_RatePM25'] >= 17)]

df3 = df3[(df3['Increase_RatePM25'] < 17) ]

df4cook = df4[(df4['Increase_RatePM25'] < 17) ]


acc 0.98
balanced acc 0.991
swsta predict
Pipeline(steps=[('resample', BorderlineSMOTE(random_state=10)),
                ('model',
                 RandomForestClassifier(max_depth=5, min_samples_leaf=10,
                                        random_state=1))])
df1 =     df1[(df1['STDPM25'] >= 35) ]
df5 =     df5[(df5['STDPM25'] >= 35) ]
df4smok = df4[(df4['STDPM25'] >= 35)]

df3 = df3[(df3['Increase_RatePM25'] < 17) ]

df4cook = df4[(df4['Increase_RatePM25'] < 17) ]

'''

df1 =     df1[(df1['STDPM25'] >= 35) ]
df5 =     df5[(df5['STDPM25'] >= 35) ]
df4smok = df4[(df4['STDPM25'] >= 35)]

df3 = df3[(df3['Increase_RatePM25'] < 17) ]

df4cook = df4[(df4['Increase_RatePM25'] < 17) ]


print(df1.shape)
print(df4smok.shape)
print(df5.shape)
print(df3.shape)

df1['Target'] = 'Smoking'
df5['Target'] = 'Smoking'
df3['Target'] = 'Cooking'
df4smok['Target'] = 'Smoking'

df4cook['Target'] = 'Cooking'











glued_data = pd.DataFrame()
glued_data = pd.concat([glued_data, df1], axis=0)
glued_data = pd.concat([glued_data, df5], axis=0)
glued_data = pd.concat([glued_data, df3], axis=0)
glued_data = pd.concat([glued_data, df4smok], axis=0)
# glued_data = pd.concat([glued_data, df4norm], axis=0)
glued_data = pd.concat([glued_data, df4cook], axis=0)
glued_data = glued_data[~glued_data.isin([np.nan, np.inf, -np.inf]).any(1)]


g0 = sns.lmplot('NSTDPM2', 'STDPM25', glued_data, hue='Target', fit_reg=False)
g0 = sns.lmplot('NSTDPM2', 'Increase_RatePM25', glued_data, hue='Target', fit_reg=False)



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






# scaler = MinMaxScaler(feature_range=(0, 10))
scaler = Normalizer(norm='l2')
# scaler = Normalizer(norm='l1')

# scaler = StandardScaler(with_mean=False)
# scaler = MaxAbsScaler()


import joblib

rand = ExtraTreesClassifier(n_estimators=50, random_state=42)  # RandomForestClassifier(n_estimators=100)
clf = rand.fit(X, y)
joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbRandFFimpo11.pkl')
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



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)



from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MaxAbsScaler

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]


hyperF = dict(model__n_estimators=n_estimators,
              model__max_depth=max_depth,
              model__min_samples_split=min_samples_split,
              model__min_samples_leaf=min_samples_leaf)



sme = SMOTEENN(smote=SMOTE(k_neighbors=3),sampling_strategy='all', random_state=42)

oversample = BorderlineSMOTE(random_state=10)
#oversample = SVMSMOTE(random_state=10)
over = ADASYN(random_state=10)

model = Pipeline([  # ('scaler',Normalizer('l2')),
    ('resample', oversample),
    ('model', RandomForestClassifier(random_state=1))
])

kf = StratifiedKFold(n_splits=3, shuffle=True)
grid_search = GridSearchCV(
    estimator=model, param_grid=hyperF, cv=kf,scoring='balanced_accuracy', refit=True,verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Adding below in as could be helpful to know how to get fitted scaler if used
best = grid_search.best_estimator_
'''RandomForestClassifier(max_depth=15, min_samples_split=10,
                                        n_estimators=300, random_state=1))])'''
print(best)

# scal = best['scaler']
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
joblib.dump(best, 'C://Users//giorgos//Desktop//AIR_quallityjobilbRandwithff11pipeline.pkl')

from sklearn.metrics import accuracy_score

print("Accuracy:{}".format(accuracy_score(y_test, pred)))
score = balanced_accuracy_score(y_test, pred)



print(f"Balanced accuracy score of a dummy classifier: {score:.3f}")



fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = r"Learning Curves (Knn, Neibors=7)"
plot_learning_curve1(best, title, X, y, axes=axes[:, 0], ylim=(0.7, 1.01),
                     cv=3, n_jobs=-1)
plot_multiclass_roc(best, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()
