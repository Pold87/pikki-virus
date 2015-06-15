import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

import numpy as np

from sklearn.preprocessing import scale

# Load datasets
df_train = pd.read_csv('TRAIN/train_hex_pred_II.csv')
df_comp = pd.read_csv('COMP/comp_hex_pred_II.csv')
#df_test = pd.read_csv('TEST/test_hex_pred_II.csv')

# create counter column
df_train['counter'] = 1

# create X_dfs
X_train = df_train.copy()
X_comp = df_comp.copy()
#X_test = df_test.copy()

species_encoder = preprocessing.LabelEncoder()

X_comp.Species[X_comp.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'
#X_test.Species[X_test.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'

groups = ['Date','HexCell','Species']
tmp = X_train.groupby(groups)['NumMosquitos'].sum()

X_train_tmp = X_train.join(X_train.groupby(groups)['NumMosquitos'].sum(), on= groups,rsuffix='_counter')

X_train = X_train_tmp.copy()
y_train = X_train.NumMosquitos_counter

# define columns to be dropped from X_dfs
droppers_train = ['Date', 'NumMosquitos', 'NumMosquitos_counter', 'WnvPresent', 'counter', 'HexCell', 'CodeSum']
droppers_comp = ['Date', 'HexCell', 'CodeSum', 'NumMosquitos']
#droppers_test = ['Id', 'Date', 'HexCell', 'CodeSum']

X_train = X_train.drop(droppers_train,axis=1)
X_comp = X_comp.drop(droppers_comp,axis=1)
#X_test = X_test.drop(droppers_test,axis=1)

X_train.Species = species_encoder.fit_transform(X_train.Species)
X_comp.Species = species_encoder.transform(X_comp.Species)
#X_test.Species = species_encoder.transform(X_test.Species)

clf = RandomForestClassifier(n_estimators = 1000)

X_train = scale(X_train)
X_comp = scale(X_comp)
#X_test = scale(X_test)

clf.fit(X_train,y_train)

y_comp = clf.predict_proba(X_comp)[:,1]
y_comp[y_comp<0]=0
#y_test = clf.predict(X_test)

df_comp['WnvPresent'] = y_comp

df_comp.to_csv("COMP/comp_hex_wnv_I.csv", index=False)
