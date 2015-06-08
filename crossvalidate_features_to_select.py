import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn import cross_validation, preprocessing, metrics
from sklearn import neighbors
import xgbwrapper
import random

import statsmodels.api as sm

from sklearn.utils import shuffle
from sklearn import linear_model
from sklearn import gaussian_process
from sklearn.linear_model import LogisticRegression

reload(xgbwrapper)

# Cross validation or submission?
do_cross_val = False

# Read in kaggle files
train = pd.read_csv("Data_with_NumMosquitos/unique_train.csv")
test = pd.read_csv("Data_with_NumMosquitos/new_test_priors.csv")

submission = pd.read_csv("sampleSubmission.csv")

features_to_select = [
    'ResultDir',
    'AvgSpeed',
    'precip_week',
    'WnvPresent_conditional_Species',
    'WnvPresent_conditional_AddressAccuracy',
    'WnvPresent_conditional_Calendar_Week',
    'WnvPresent_conditional_Tmin',
    'WnvPresent_conditional_ResultDir',
    'Species',
    'Latitude',
    'Longitude',
    'precip_week',
    '5_week_avrgPrecipTotal',
    '6_week_avrgPrecipTotal',
    '7_week_avrgPrecipTotal',
    '5_week_avrgTavg',
    '6_week_avrgTavg',
    '7_week_avrgTavg',
    '8_week_avrgTavg',
    '9_week_avrgTavg',
    'Year']

# Training data frame for leave-one-year-out CV
train_for_loo = train[features_to_select  + ['WnvPresent_DateTrapSpecies']]

X = train[features_to_select]

# Test data set
X_real_test = test[features_to_select]

def year_train_test_split(train, target, year):

    # Create copy
    X = train.copy()

    # Retrieve target column and remove from X
    y = X[target]
    X.drop([target], axis=1)

    # Create mask
    msk = X.Year == year

    # Drop date column
    X = X.drop(['Year',
                #'Date',
                'WnvPresent_DateTrapSpecies'], axis=1)
    

    # Create dfs based on mask    
    X_train = X[~msk]
    X_test = X[msk]
    y_train = y[~msk]
    y_test = y[msk]

    return X_train, X_test, y_train, y_test


# Create classifiers

clf = RandomForestClassifier(n_estimators=500,
                            min_samples_leaf=5)

clf = xgbwrapper.XgbWrapper({'objective': 'binary:logistic',
                             'eval_metric': 'auc',
                             'eta': 0.05,
                             'silent': 1})


# Cross validation
if do_cross_val:

    # Leave-one-year-out cross-validation
    scores = []
    total_pred = np.array([])
    total_test = np.array([])
    
    for year in [2007, 2009, 2011, 2013]:

        X_train, X_test, y_train, y_test = year_train_test_split(
            train_for_loo,
            'WnvPresent_DateTrapSpecies',
            year)      

        X_train.to_csv("data_per_year/" + str(year) + "X_train.csv", index=False)
        X_test.to_csv("data_per_year/" + str(year) + "X_test.csv", index=False)
        y_train.to_csv("data_per_year/" + str(year) + "y_train.csv", index=False)
        y_test.to_csv("data_per_year/" + str(year) + "y_test.csv", index=False)

        
        clf.fit(X_train, y_train)

        # y_pred = clf.predict_proba(X_test) [:, 1] # Random Forest
        y_pred = clf.predict_proba(X_test) # For XGB
        
        score = metrics.roc_auc_score(y_test, y_pred)
        scores.append(score)
        
        #import operator
        #feat_importances = dict(zip(X_train.columns, clf.feature_importances_))
        #sorted_feat_importances = sorted(feat_importances.items(), key=operator.itemgetter(1))
        #print(sorted_feat_importances)
        
        total_pred = np.concatenate((total_pred, y_pred))
        total_test = np.concatenate((total_test, y_test))
        
    print("Global ROC score", metrics.roc_auc_score(total_test, total_pred))
        
    print(scores)
    print(np.array(scores).mean())

# Make submission

else:
    clf.fit(X, train.WnvPresent_DateTrapSpecies)


    # y = clf.predict_proba(X_real_test) [:, 1] # Random Forest
    y = clf.predict_proba(X_real_test) # For XGB 

    submission.WnvPresent = y
    submission.to_csv("unqiue_XGB.csv", index=False)
