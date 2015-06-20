import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import os.path

import numpy as np

from sklearn.preprocessing import scale

# Load datasets
df_train = pd.read_csv('TRAIN/train_hex_wnv_pred_I.csv')
df_test = pd.read_csv('TEST/test_hex_wnv_pred_I.csv')
submission = pd.read_csv("SUBMISSION/sampleSubmission.csv")

# create counter column
df_train['counter'] = 1

# create X_dfs
X_train = df_train.copy()
X_test = df_test.copy()

species_encoder = preprocessing.LabelEncoder()

X_test.Species[X_test.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'

X_train = X_train.copy()
y_train = X_train.WnvPresent

# define columns to be dropped from X_dfs
droppers_train = ['Date', 'NumMosquitos', 'WnvPresent', 'counter', 'HexCell', 'CodeSum']
droppers_test = ['Id', 'Date', 'HexCell', 'CodeSum']

X_train = X_train.drop(droppers_train,axis=1)
X_test = X_test.drop(droppers_test,axis=1)

X_train.Species = species_encoder.fit_transform(X_train.Species)
X_test.Species = species_encoder.transform(X_test.Species)

clf = RandomForestClassifier(n_estimators = 1000)

X_train = scale(X_train)
X_test = scale(X_test)

clf.fit(X_train,y_train)

y_test = clf.predict_proba(X_test)[:,1]

y_test[y_test<0]=0
#y_test = clf.predict(X_test)

df_test['WnvPresent'] = y_test

df_test.to_csv("COMP/comp_hex_wnv_II.csv", index=False)

submission.WnvPresent = y_test
i=0
written = False
while not written:
    i += 1
    file_path = os.getcwd() + "/SUBMISSION/hex_submission_{}.csv".format(i)
    if not os.path.exists(file_path):
        submission.to_csv(file_path, index=False)
        written = True
