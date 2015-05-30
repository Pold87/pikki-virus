import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn import cross_validation, preprocessing, metrics
from xgbwrapper import XgbWrapper

do_cross_val = 2

# Read in kaggle files
train = pd.read_csv("train_filled.csv")
test = pd.read_csv("test_filled.csv")
submission = pd.read_csv("sampleSubmission.csv")


train_for_loo = train.drop(['NumMosquitos',
                          'AddressAccuracy',
                          'AddressNumberAndStreet',
                          'Address',
                          'Street'], axis=1)

X = train.drop(['Date',
                'NumMosquitos',
                'AddressAccuracy',
                'AddressNumberAndStreet',
                'Address',
                'Street',
                'WnvPresent'], axis=1)

X_real_test = test.drop(['Id',
                         'Date',
                         'Address',
                         'Street',
                         'AddressAccuracy',
                         'AddressNumberAndStreet'], axis=1)

species_encoder = preprocessing.LabelEncoder()
trap_encoder = preprocessing.LabelEncoder()

X.Species = species_encoder.fit_transform(X.Species)
X.Trap = trap_encoder.fit_transform(X.Trap)

train_for_loo.Species = species_encoder.fit_transform(train_for_loo.Species)
train_for_loo.Trap = trap_encoder.fit_transform(train_for_loo.Trap)


X_real_test.Species = species_encoder.fit_transform(X_real_test.Species)
X_real_test.Trap = species_encoder.fit_transform(X_real_test.Trap)


def get_year(dt):
    return int(str.split(dt, '-')[0])


def year_train_test_split(train, target, year):

    # Create copy
    X = train.copy()

    # Retrieve target column and remove from X
    y = X[target]
    X.drop([target], axis=1)

    # Create year column
    X['year'] = X.Date.apply(get_year)

    # Create mask
    msk = X.year == year

    # Drop date column
    X = X.drop(['year', 'Date', 'WnvPresent'], axis=1)
    

    # Create dfs based on mask    
    X_train = X[~msk]
    X_test = X[msk]
    y_train = y[~msk]
    y_test = y[msk]

    return X_train, X_test, y_train, y_test


# Cross validation

# Create classifier
clf = GradientBoostingClassifier(n_estimators=1000,
                                 random_state=35,
                                 min_samples_leaf=6)

clf = XgbWrapper({'objective': 'binary:logistic',
                  'eval_metric': 'auc',
                  'eta': 0.1,
                  'max_delta_step': 1})


# 'Normal' 70 / 30 cross-validation
if do_cross_val == 1:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        train.WnvPresent,
        test_size=0.3,
        random_state=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)#[:, 1]
    print(metrics.roc_auc_score(y_test, y_pred))

elif do_cross_val == 2:

    # Leave-one-year-out cross-validation
    scores = []
    for year in [2007, 2009, 2011, 2013]:

        X_train, X_test, y_train, y_test = year_train_test_split(
            train_for_loo,
            'WnvPresent',
            year)
        
        
        clf.fit(X_train, y_train)

        y_pred = clf.predict_proba(X_test)
        score = metrics.roc_auc_score(y_test, y_pred)
        scores.append(score)
    print(scores)

else:
    clf.fit(X, train.WnvPresent)

    # Make submission
    y = clf.predict_proba(X_real_test)[:, 1]
    submission.WnvPresent = y
    submission.to_csv("ourSub.csv", index=False)
