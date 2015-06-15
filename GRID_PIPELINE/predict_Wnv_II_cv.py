import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing

from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

from sklearn.preprocessing import scale

from time import time


df_train = pd.read_csv('TRAIN/train_hex_weather.csv')
df_test = pd.read_csv('TEST/test_hex_weather.csv')
df_comp = pd.read_csv('COMP/comp_hex_weather.csv')

df_train['counter'] = 1

X_train = df_train.copy()
X_test = df_test.copy()
X_comp = df_comp.copy()

species_encoder = preprocessing.LabelEncoder()

X_test.Species[X_test.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'
X_comp.Species[X_comp.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'

groups = ['Date','HexCell','Species']
tmp = X_train.groupby(groups)['NumMosquitos'].sum()

X_train_tmp = X_train.join(X_train.groupby(groups)['NumMosquitos'].sum(), on= groups,rsuffix='_counter')

y = X_train_tmp.NumMosquitos_counter

del X_train_tmp['Date']
del X_train_tmp['NumMosquitos']
del X_train_tmp['NumMosquitos_counter']
del X_train_tmp['WnvPresent']
del X_train_tmp['counter']
del X_train_tmp['HexCell']
del X_train_tmp['CodeSum']

X_train_tmp.Species = species_encoder.fit_transform(X_train.Species)
X_test.Species = species_encoder.transform(X_test.Species)
X_comp.species = species_encoder.transform(X_comp.Species)

X_train_tmp = scale(X_train_tmp)

clf_list= [#RandomForestRegressor(n_estimators=100,n_jobs=-1),
           #RandomForestRegressor(n_estimators=400,n_jobs=-1),
           #RandomForestRegressor(n_estimators=1000,n_jobs=-1), 
           #Ridge(alpha = .01,normalize=True),
           #Ridge(alpha = .1,normalize=True),
           #Ridge(alpha = .1),
           #GradientBoostingRegressor(n_estimators = 100,max_depth=7),
          # GradientBoostingRegressor(n_estimators = 1000,max_depth=10),
           #GradientBoostingRegressor(n_estimators = 1000,max_depth=15,subsample = .7),#so far the best one, n_es = 400
           GradientBoostingRegressor(n_estimators = 100,max_depth=15,learning_rate = 0.01),
           GradientBoostingRegressor(n_estimators = 100,max_depth=15,subsample = .7),
           GradientBoostingRegressor(n_estimators = 400,max_depth=15,learning_rate = 0.01),
           GradientBoostingRegressor(n_estimators = 400,max_depth=15,subsample = .7),
           GradientBoostingRegressor(n_estimators = 1000,max_depth=15,learning_rate = 0.01),
           GradientBoostingRegressor(n_estimators = 1000,max_depth=15,subsample = .7),
          # GradientBoostingRegressor(n_estimators = 1000,max_depth=20,subsample = .7)
           #KNeighborsRegressor(),
           #SGDRegressor()
           #GradientBoostingRegressor(n_estimators = 100,max_depth=4),
           #GradientBoostingRegressor(n_estimators = 100,max_depth=2)
      ]
random_state= 46

err = []
r2 = []
#get lengths of y_arr -.-

X_train, X_test,y_train, y_test = train_test_split(X_train_tmp,y,test_size=.25, random_state= random_state)

y_arr = pd.DataFrame(y_test)
i = 1
#clf = clf_list[1]
for clf in clf_list:
#for i in np.random.randint(1,size=4):
    i += 1
    X_train, X_test,y_train, y_test = train_test_split(X_train_tmp,y,test_size=.25, random_state= random_state)
    t = time()
    print('Starting training')
    clf.fit(X_train,y_train)
    print('End training after: ', int((time()-t)/60.0))
    yhat = clf.predict(X_test)
    err.append( mean_squared_error(yhat,y_test))
    r2.append(r2_score(yhat,y_test))
    y_arr[i] = yhat

y_arr_mean = y_arr.drop(0,axis=1)
y_arr_mean = y_arr_mean.mean(axis=1)
print('Mean',y_arr_mean)
print(r2_score(y_arr_mean,y_arr[0]))
print(r2)
#y_train = df_train['NumMosquitos']
#
#clf.fit(X_train,y_train)


