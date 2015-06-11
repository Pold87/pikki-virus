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

do_cross_val = 2

predict_num_mosquitos = False

# Read in kaggle files
train = pd.read_csv("volker_unique_train.csv")
test = pd.read_csv("test_filled_new.csv")
submission = pd.read_csv("sampleSubmission.csv")


# Dummy variables
s = pd.Series(list(train.Species))
dummies = pd.get_dummies(s)
train.drop('Species', axis=1, inplace=True)
train['UNSPECIFIED SPECIES'] = 0

# Create df for leave-one-year-out cross-validation
train_for_loo = train.drop([#'NumMosquitos',
                            'Trap',
                            'Month',
                            'Block',
                            'AddressAccuracy',
                            'AddressNumberAndStreet',
                            'Address',
                            'Street',
                            'CodeSum'
], axis=1)

# train_for_loo = train[features_to_select  + ['WnvPresent']]

# Create df for training on the full training set
X = train.drop(['Date',
                'Month',
                'Trap',
                'Block',
                # 'NumMosquitos',
                'AddressAccuracy',
                'AddressNumberAndStreet',
                'Address',
                'Street',
                'CodeSum',
                'WnvPresent'], axis=1)

# X = train[features_to_select]

# Dummy variables
s = pd.Series(list(test.Species))
dummies = pd.get_dummies(s)
test.drop('Species', axis=1, inplace=True)
pd.concat([dummies, test], axis=1)

# Create df for testing and predicting
X_real_test = test.drop(['Id',
                         'Date',
                         'Month',
                         'Block',
                         'Trap',
                         'Address',
                         'Street',
                         'AddressAccuracy',
                         'CodeSum',
                         'AddressNumberAndStreet'], axis=1)

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
    X = X.drop(['Year', 'Date', 'WnvPresent', 'Id'], axis=1)
    
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

        
    
# Create classifier
#clf = GradientBoostingClassifier(n_estimators=100,
#                                 random_state=35,
#                                 min_samples_leaf=6)
#
clf = RandomForestClassifier(n_estimators=1,
                             min_samples_leaf=4)

# clf = neighbors.KNeighborsClassifier(50,
#                                      p=3)

# clf = neighbors.KernelDensity()

# clf = LogisticRegression()
# clf = linear_model.Ridge(alpha=0.1)
#clf = linear_model.BayesianRidge(n_iter=5000,
#                                 normalize=True)

#clf = linear_model.ARDRegression(n_iter=500,
#                                 normalize=True)

#clf = xgbwrapper.XgbWrapper({'objective': 'binary:logistic',
#                  'eval_metric': 'auc',
#                  'eta': 0.1,
#                  'silent': 0,
#                  'max_delta_step': 1})


# 'Normal' 70 / 30 cross-validation
if do_cross_val == 1:
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X,
        train.WnvPresent,
        test_size=0.3,
        random_state=0)

    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)[:, 1]
    print(metrics.roc_auc_score(y_test, y_pred))

elif do_cross_val == 2:

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

        y_pred = clf.predict_proba(X_test) [:, 1]
        # print(y_pred)
              
        # y_pred = clf.predict_proba(X_test) # For xgbwrapper best score: 57.2
        #         y_pred = clf.predict_proba(X_test)
        # y_pred = clf.predict(X_test)

        score = metrics.roc_auc_score(y_test, y_pred)
        scores.append(score)

        import operator
        feat_importances = dict(zip(X_train.columns, clf.feature_importances_))
        sorted_feat_importances = sorted(feat_importances.items(), key=operator.itemgetter(1))
        
        print(sorted_feat_importances)
        
        total_pred = np.concatenate((total_pred, y_pred))
        total_test = np.concatenate((total_test, y_test))

    #for x, y in zip(total_test, total_pred):
    #    print(x, y, x-y)
        
    print("Global ROC score", metrics.roc_auc_score(total_test, total_pred))
        
    print(scores)
    print(np.array(scores).mean())

elif do_cross_val == 3:

    # TODO:
    # Implement lasagne

    
    # Leave-one-year-out cross-validation
    scores = []
    for year in [2007, 2009, 2011, 2013]:

        X_train, X_test, y_train, y_test = year_train_test_split(
            train_for_loo,
            'WnvPresent',
            year)

        print(X_test.head())
        
        X_train = np.asarray(X_train, dtype=np.float32)
        X_test = np.asarray(X_test, dtype=np.float32)

        y_train = np.asarray(y_train, dtype=np.int32).reshape(-1,1)
        y_test = np.asarray(y_test, dtype=np.int32).reshape(-1,1)

        input_size = len(X_train[0])

        learning_rate = theano.shared(np.float32(0.1))

        clf = NeuralNet(
            layers=[  
                ('input', InputLayer),
                ('hidden1', DenseLayer),
                ('dropout1', DropoutLayer),
                ('hidden2', DenseLayer),
                ('dropout2', DropoutLayer),
		('hidden3',DenseLayer),
		('dropout3', DemseLayer),
                ('output', DenseLayer),
            ],
            # layer parameters:
            input_shape=(None, input_size), 
            hidden1_num_units=256, 
            dropout1_p=0.5,
            hidden2_num_units=150, 
            dropout2_p=0.4,
	    hidden3_num_units = 100,
	    dropout3_p = 0.1,
            output_nonlinearity=sigmoid, 
            output_num_units=1, 
            
            # optimization method:
            update=nesterov_momentum,
            update_learning_rate=learning_rate,
            update_momentum=0.9,
            
            # Decay the learning rate
            on_epoch_finished=[
                AdjustVariable(learning_rate, target=0, half_life=4),
            ],
            
            # This is silly, but we don't want a stratified K-Fold here
            # To compensate we need to pass in the y_tensor_type and the loss.
            regression=True,
            y_tensor_type = T.imatrix,
            objective_loss_function = binary_crossentropy,
            
            max_epochs=1, 
            eval_size=0.2,
            verbose=1,
        )

        X, y = shuffle(X_train, y_train, random_state=123)
        
        clf.fit(X, y)


        _, X_valid, _, y_valid = clf.train_test_split(X, y, clf.eval_size)
        probas = clf.predict_proba(X_valid)[:,0]

        #y_pred = clf.predict_proba(X_test)[:, 0]

        #print(y_pred)
        
        #score = metrics.roc_auc_score(y_test, y_pred)
        #scores.append(score)
    print("Global ROC score", metrics.roc_auc_score(toal_valid, total_probas))
    print(scores)    
    
else:
    clf.fit(X, train.WnvPresent)

    # Make submission
    y = clf.predict_proba(X_real_test)[:, 1]
    submission.WnvPresent = y
    submission.to_csv("ourSub_mvgAvgs.csv", index=False)
