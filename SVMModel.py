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
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, balanced_accuracy_score, \
    average_precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import resample

from Optunity import plot_learning_curve, plot_multiclass_roc, ffImportance

'''print(sklearn.__version__)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

path2 = "G://sensor smoking//Feutures//"
path1 = "G://sensor_normal//Feutures//"

name1 = 'Feautures_AA_normal_1.csv'
name2 = 'Feautures_AA_Smok_1.csv'

name1 = 'Feautures_AA_normal_1.csv'
name2 = 'Feautures_AA_Smok_1.csv'

path3 = "G://sensor_cooking//Feautures//"
name3 = 'Feautures_AA_Cooking_1.csv'

path1 = "G://MyData//Normal//"
path2 = "G://MyData//Smoking//"
path3 = "G://MyData//Cooking//"

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

df2 = dfNormal  # pd.read_csv(path1 + name1)
df1 = pd.read_csv(path2 + name2)
df3 = dfCooking  # pd.read_csv(path3 + name3)

df2 = df2[(df2['NSTDPM2'] < 0.3) & (df2['Increase_RatePM25'] <= 0.3)]  # normal
df1 = df1[(df1['NSTDPM2'] > 0.3) & (df1['Increase_RatePM25'] >= 2.5)]
df3 = df3[(df3['NSTDPM2'] > 0.3) & (df3['Increase_RatePM25'] <= 2.4) & (df3['Increase_RatePM25'] >= 0.3)]


#df1['Target'] = df1['NSTDPM2'].apply(lambda x: 'Smoking' if x > 0.3 else 'normal')

#df3['Target'] = df3['NSTDPM2'].apply(lambda x: 'Cooking' if (x > 0.3) else 'normal')


print(df1.shape)
print(df3.shape)
print(df2.shape)

df1['Target'] = 'Smoking'
df2['Target'] = 'normal'
df3['Target'] = 'Cooking'
#


glued_data = pd.DataFrame()
glued_data = pd.concat([glued_data, df1], axis=0)
glued_data = pd.concat([glued_data, df2], axis=0)
glued_data = pd.concat([glued_data, df3], axis=0)
le = preprocessing.LabelEncoder()
le.fit(glued_data[['Target']])
glued_data['Target'] = le.transform(glued_data[['Target']])
print(glued_data)
'''
# glued_data['Target']=le.inverse_transform(glued_data[['Target']])
# print(glued_data)
'''
glued_data = glued_data.drop('NSTDPM2', axis=1)
glued_data = glued_data.drop('Unnamed: 0', axis=1)
glued_data = glued_data.drop(['STDPM1',
                              'STDPM25', 'STDPM10', 'STDHumidity', 'STDTemp', 'CMagHUmidity',
                              'CMagTemperature', 'ChangeRate1', 'ChangeRate25', 'ChangeRate10', 'ChangeMagn1',
                              'ChangeMagn25',
                              'ChangeMagn10', 'ChangeIncreaseMagn_rate1', 'ChangeIncreaseMagn_rate25',
                              'ChangeIncreaseMagn_rate10', 'ChangeDecreaseMagn_rate1',
                              'ChangeDecreaseMagn_rate25', 'ChangeDecreaseMagn_rate10', 'STDRATIOrate1',
                              'STDRATIOrate25',
                              'STDRATIOrate10'], axis=1)
print(glued_data)
y = glued_data['Target']
X = glued_data.drop('Target', axis=1)

# Separate majority and minority classes
df_majority = glued_data[glued_data['Target'] == 0]
df_minority = glued_data[glued_data['Target'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=415,  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled = df_minority_upsampled  # pd.concat([df_majority, df_minority_upsampled])

df_minority2 = glued_data[glued_data['Target'] == 2]

# Upsample minority class
df_minority1_upsampled = resample(df_minority2,
                                  replace=True,  # sample with replacement
                                  n_samples=415,  # to match majority class
                                  random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled1 = pd.concat([df_upsampled, df_minority1_upsampled])

df_minority3 = glued_data[glued_data['Target'] == 0]

# Upsample minority class
df_minority3_upsampled = resample(df_minority3,
                                  replace=True,  # sample with replacement
                                  n_samples=415,  # to match majority class
                                  random_state=123)  # reproducible results

# Combine majority class with upsampled minority class
df_upsampled3 = pd.concat([df_upsampled1, df_minority3_upsampled])

# Display new class counts
print(df_upsampled3['Target'].value_counts())
y = df_upsampled3['Target']
X = df_upsampled3.drop('Target', axis=1)
print('edw')
print(X)
print(y)
'''

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

''''''
df2 = df2[(df2['NSTDPM2'] < 0.3) & (df2['Increase_RatePM25'] <= 0.3) & (df2['STDPM25'] <= 5)]  # normal
df1 = df1[(df1['NSTDPM2'] > 0.3) & (df1['Increase_RatePM25'] >= 2.5)]
df3 = df3[(df3['NSTDPM2'] > 0.3) & (df3['Increase_RatePM25'] <= 2.5) & (df3['Increase_RatePM25'] >= 0.3)]

df4cook = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] <= 2.5)]

df4smok = df4[(df4['NSTDPM2'] > 0.3) & (df4['Increase_RatePM25'] >= 2.5)]

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
print(glued_data)
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

ffImportance(clf,X)



import joblib

joblib.dump(clf, 'C://Users//giorgos//Desktop//AIR_quallityjobilbSVMFFimpo.pkl')
sel = SelectFromModel(clf, prefit=True)
X = sel.transform(X)



# scaler = MinMaxScaler(feature_range=(0, 2))
scaler = Normalizer(norm='l2')
# scaler = Normalizer(norm='l1')
# scaler = StandardScaler(with_std=False)
# scaler = MaxAbsScaler()
scaler.fit(X)
rescaledX=scaler.transform(X)
joblib.dump(scaler, 'C://Users//giorgos//Desktop//AIR_quallityjobilb_SVMFF12scaler.pkl')
X=rescaledX

training_features, test_features, training_target, test_target, = train_test_split(X, y, test_size=.1, random_state=12)
strategy = {0: 800, 1: 800}
sm = SMOTE(sampling_strategy='all', k_neighbors=5, random_state=10)

x_res, y_res = sm.fit_resample(training_features, training_target)
print(np.bincount(y_res))
# print(training_target.value_counts())


X = x_res
y = y_res
# function(X, 'sc')


strategy = {0: 200, 1: 200}
sm1 = SMOTE(sampling_strategy='all', k_neighbors=1, random_state=10)

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


print(X.shape)
# ALgorithm


kernels = ['Polynomial', 'RBF', 'Sigmoid', 'Linear']  # A function which returns the corresponding SVC model


def getClassifier(ktype):
    if ktype == 0:
        # Polynomial kernal
        return SVC(kernel='poly', degree=8, gamma="auto")
    elif ktype == 1:
        # Radial Basis Function kernal
        return SVC(kernel='rbf', gamma="auto")
    elif ktype == 2:
        # Sigmoid kernal
        return SVC(kernel='sigmoid', gamma="auto")
    elif ktype == 3:
        # Linear kernal
        return SVC(C=10, kernel='linear', gamma=1)


cv_scores = []
for i in range(4):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)
    #svclassifier = getClassifier(i)
    # scores = cross_val_score(svclassifier, X, y, cv=10)
    '''

    svclassifier.fit(X_train, y_train)  # Make prediction
    y_pred = svclassifier.predict(X_test)  # Evaluate our model
    print("Evaluation:", kernels[i], "kernel")
    print(classification_report(y_test, y_pred))
    '''
    # cv_scores.append(scores.mean())
    # print(cv_scores)

# https://github.com/clareyan/SVM-Hyper-parameter-Tuning-using-GridSearchCV/blob/master/SVM_Iris.ipynb
# https://elitedatascience.com/imbalanced-classes


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

def GridFunc():
    from sklearn.metrics import accuracy_score
    # tuning hyperparamaters for SVM
    # X_train, X_test, y_train, y_test = train_test_split(rescaledX, y, test_size=0.2, random_state=42)
    param_grid = {'C': [0.1, 10, 25, 50, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['linear', 'rbf', 'sigmoid']}

    parameters = {
        "estimator__C": [0.1, 10, 25, 50, 100, 1000],
        "estimator__kernel": ['linear', 'rbf', 'sigmoid', 'poly'],
        "estimator__gamma": [1, 0.1, 0.01, 0.001, 0.0001]

    }
    scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
    # model_to_set = OneVsRestClassifier(SVC())

    #svm=SVC(decision_function_shape='ovr')#decision_function_shape='ovr'
    #svm.fit(X_train,y_train)
    grid = GridSearchCV(SVC(), param_grid=param_grid, cv=5, scoring='accuracy', refit=True, verbose=2)

    grid.fit(X_train, y_train)

    grid_predictions = grid.predict(test_features)

    print(confusion_matrix(test_target, grid_predictions))
    print(classification_report(test_target, grid_predictions))
    print('best')
    print(np.unique(grid_predictions))
    # print best parameter after tuning
    print(grid.best_params_)

    # print how our model looks after hyper-parameter tuning
    print(grid.best_estimator_)
    from sklearn.metrics import accuracy_score
    print("Accuracy edw:{}".format(accuracy_score(y_test, grid_predictions)))
    cm = confusion_matrix(test_target, grid_predictions)
    df = cm2df(cm, le.classes_)
    print(df)

    '''
    

    y_pred = svclassifier.predict(X_test)

    from sklearn.metrics import accuracy_score

    print("Accuracy edw:{}".format(accuracy_score(y_test, y_pred)))

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    df = cm2df(cm, le.classes_)
    print(df)

'''








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

svm=SVC(C=1000, gamma= 1, kernel='rbf',probability=True)#SVC(**grid.best_params_)
svm.fit(X_train,y_train)
pred=svm.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy edw:{}".format(accuracy_score(y_test, pred)))
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred)
df = cm2df(cm, le.classes_)
print(df)


fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = r"Learning Curves (Random forest, Neibors=7)"

plot_learning_curve(svm, title, X_train, y_train, X_test, y_test, axes=axes[:, 0], ylim=(0.7, 1.01),
                    cv=10, n_jobs=-1)
plt.show()


# now will try the same with joblib
import joblib

joblib.dump(svm, 'C://Users//giorgos//Desktop//AIR_quallityjobilb_SVMFF.pkl')

# load and test this

whatever = joblib.load('C://Users//giorgos//Desktop//AIR_quallityjobilb_SVM1.pkl')

# Ypredict = whatever.predict(X_test)
# print(Ypredict)
print(joblib.__version__)
score = balanced_accuracy_score(y_test, pred)
print(f"Balanced accuracy score of a dummy classifier: {score:.3f}")
plot_multiclass_roc(svm, X_test, y_test, n_classes=2, figsize=(16, 10))
plt.show()

print(classification_report(y_test, pred))
best=svm
probs = best.decision_function(X_test)
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
