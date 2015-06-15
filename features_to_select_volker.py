import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble.forest import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn import cross_validation, preprocessing, metrics
from sklearn import neighbors
import xgbwrapper
import random

import statsmodels.api as sm

from lasagne.layers import InputLayer, DropoutLayer, DenseLayer
from lasagne.updates import nesterov_momentum
from lasagne.objectives import binary_crossentropy
from nolearn.lasagne import NeuralNet
import theano
from theano import tensor as T
from theano.tensor.nnet import sigmoid
from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.linear_model import LogisticRegression

reload(xgbwrapper)

do_cross_val = True
predict_num_mosquitos = False
make_sub = False
drop = 1

# Read in kaggle files
train = pd.read_csv("volker_unique_train.csv")
#train = pd.read_csv("train_filled_new.csv")
test = pd.read_csv("test_filled_new.csv")

# Dummy variables
s = pd.Series(list(train.Species))
dummies = pd.get_dummies(s)
train.drop('Species', axis=1, inplace=True)

train = pd.concat([dummies, train], axis=1)


submission = pd.read_csv("sampleSubmission.csv")
        

features_to_select = [
    'Species',
    'Latitude',
    'Longitude',
    'precip_week',
    'Tavg_1_weeks_ago',
    'Tavg_2_weeks_ago',
    'Tavg_3_weeks_ago',
    'PrecipTotal_1_weeks_ago',
    'PrecipTotal_2_weeks_ago',
    'PrecipTotal_3_weeks_ago',
    '5_week_avrgPrecipTotal',
    '6_week_avrgPrecipTotal',
    '7_week_avrgTavg',
    '8_week_avrgTavg',
    '9_week_avrgTavg',
#     '10_week_avrgTavg',
    'Year',
    'Date']


# Create df for leave-one-year-out cross-validation

train_for_loo = train[features_to_select  + ['WnvPresent',
                                              'NumMosquitos']]


# Create df for training on the full training set

X = train[features_to_select + ['NumMosquitos']]


# Create df for testing and predicting
X_real_test = test[features_to_select]

species_encoder = preprocessing.LabelEncoder()
trap_encoder = preprocessing.LabelEncoder()

X.Species = species_encoder.fit_transform(X.Species)
# X.Trap = trap_encoder.fit_transform(X.Trap)

train_for_loo.Species = species_encoder.transform(train_for_loo.Species)
# train_for_loo.Trap = trap_encoder.fit_transform(train_for_loo.Trap)

# Handle UNSPECIFIED CULEX

all_species = train.Species.unique()

unspecified_mask = X_real_test.Species == "UNSPECIFIED CULEX"

# TODO: Or use worst case!
X_real_test.ix[unspecified_mask, "Species"] = np.random.choice(all_species, len(unspecified_mask))
X_real_test.Species = species_encoder.transform(X_real_test.Species)


# X_real_test.Trap = trap_encoder.transform(X_real_test.Trap)

def year_train_test_split(train, target, year):

    # Create copy
    X = train.copy()

    # Retrieve target column and remove from X
    y = X[target]
    X.drop([target], axis=1)

    # Create mask
    msk = X.Year == year

    # Drop date column
    X = X.drop(['Year', 'Date', 'WnvPresent'], axis=1)
    
    # Create dfs based on mask    
    X_train = X[~msk]
    X_test = X[msk]
    X_test = X_test.drop(['NumMosquitos'], axis=1)
    y_train = y[~msk]
    y_test = y[msk]
    y_train_numMosquitos = X.NumMosquitos[~msk] 
    y_test_numMosquitos = X.NumMosquitos[msk] 
    
    return X_train, X_test, y_train, y_test, y_train_numMosquitos, y_test_numMosquitos


# Cross validation

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))    

        
#clf = RandomForestClassifier(n_estimators=1000,
#                             min_samples_leaf=6)

#clf = xgbwrapper.XgbWrapper({'objective': 'binary:logistic',
#                  'eval_metric': 'auc',
#                  'eta': 0.1,
#                  'silent': 0,
#                  'max_delta_step': 1})

# Leave-one-year-out cross-validation
scores = []
total_pred = np.array([])
total_test = np.array([])


for year in [2007, 2009, 2011, 2013]:

    X_train,X_test, y_train, y_test, y_train_numMosquitos, y_test_numMosquitos = year_train_test_split(
        train_for_loo,
        'WnvPresent',
        year)      

    X_train.to_csv("data_per_year/" + str(year) + "X_train.csv", index=False)
    X_test.to_csv("data_per_year/" + str(year) + "X_test.csv", index=False)
    y_train.to_csv("data_per_year/" + str(year) + "y_train.csv", index=False)
    y_test.to_csv("data_per_year/" + str(year) + "y_test.csv", index=False)


    if predict_num_mosquitos:
        reg = GradientBoostingRegressor(n_estimators=40)

        reg.fit(X_train.drop(['NumMosquitos'], axis=1), y_train_numMosquitos.astype(float))
        predicted_mosquitos = reg.predict(X_test)
        X_test['NumMosquitos'] = predicted_mosquitos
        print("Accuracy is", metrics.r2_score(y_test_numMosquitos, predicted_mosquitos))

    clf.fit(X_train.drop(['NumMosquitos'], axis=1), y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    # print(y_pred)

    # y_pred = clf.predict_proba(X_test) # For xgbwrapper best score: 57.2
    #         y_pred = clf.predict_proba(X_test)
    # y_pred = clf.predict(X_test)



    non_carriers_mask = (X_test.Species == species_encoder.transform('CULEX SALINARIUS')) |\
                        (X_test.Species == species_encoder.transform('CULEX ERRATICUS')) |\
                        (X_test.Species == species_encoder.transform('CULEX TARSALIS')) |\
                        (X_test.Species == species_encoder.transform('CULEX TERRITANS'))

    # y_pred[non_carriers_mask] = 0
    #score = metrics.roc_auc_score(y_test, y_pred)
    #scores.append(score)

    import operator
    feat_importances = dict(zip(X_train.columns, clf.feature_importances_))
    sorted_feat_importances = sorted(feat_importances.items(), key=operator.itemgetter(1))

    # print(sorted_feat_importances)

    print(y_pred)
    print(y_test)

    total_pred = np.concatenate((total_pred, y_pred))
    total_test = np.concatenate((total_test, y_test))

#for x, y in zip(total_test, total_pred):
#    print(x, y, x-y)

print("Global ROC score", metrics.roc_auc_score(total_test, total_pred))

if make_sub:
    clf.fit(X, train.WnvPresent)

    # Make submission
    y = clf.predict_proba(X_real_test)[:, 1]
    submission.WnvPresent = y
    submission.to_csv("ourSub_mvgAvgs.csv", index=False)
