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


# Read in kaggle files
train = pd.read_csv("Data_with_NumMosquitos/unique_train.csv")
test = pd.read_csv("Data_with_NumMosquitos/new_test_priors.csv")
submission = pd.read_csv("sampleSubmission.csv")


# Dummy variables for Species

# Actually creating dummy variables is even easier than that (but now
# I have it like that and don't want to ruin the code). But train =
# pd.get_dummies(train, columns=['Species']) should work as well.

## Train
s = pd.Series(list(train.Species))
dummies = pd.get_dummies(s)
train.drop(['Id',


            'Tmax',
            'Tmin',
            'Tavg',
            'SeaLevel',
            'Month',
            'Station',
            'Species',
            'SnowFall',
            'WetBulb',

            'WnvPresent_conditional_Depth',
            'WnvPresent_conditional_Station',
            'WnvPresent_conditional_SnowFall',
            'PrecipTotal',
            'cool_dw',
            'Heat',
            'Sunset',
            'Sunrise',
            'DewPoint',

#            'count_mosquitos',  # Include predicted counts
            
            'Block',
            'Year',
            'Depth'
], axis=1, inplace=True)


# S7 is unknown species
train['S7'] = 0
train = pd.concat([dummies, train], axis=1)

train = train.rename(columns={
    0: 'S0',
    1: 'S1',
    2: 'S2',
    3: 'S3',
    4: 'S4',
    5: 'S5',
    6: 'S6'})

train = train.sort_index(axis=1)

y_train = np.asarray(train.WnvPresent_DateTrapSpecies, dtype=np.int32).reshape(-1,1)

## Test
s = pd.Series(list(test.Species))
dummies = pd.get_dummies(s)

test.drop(['Id',

           
           'Tmax',
           'Tmin',
           'Tavg',
           'SeaLevel',
           'Month',
           'Station',
           'Species',
           'SnowFall',
           'WetBulb',

           'WnvPresent_conditional_Depth',
           'WnvPresent_conditional_Station',
           'WnvPresent_conditional_SnowFall',
           'PrecipTotal',
           'cool_dw',
           'Heat',
           'Sunset',
           'Sunrise',
           'DewPoint',


#           'count_mosquitos',  # Include predicted counts
           
           
           'Block',
           'Year',
           'Depth',
           
           
           
           'Unnamed: 0'], axis=1, inplace=True)

test = pd.concat([dummies, test], axis=1)

test = test.rename(columns={
    0: 'S0',
    1: 'S1',
    2: 'S2',
    3: 'S3',
    4: 'S4',
    5: 'S5',
    6: 'S6',
    7: 'S7'})

test = test.sort_index(axis=1)

class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))


def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std


train.drop(['WnvPresent_DateTrapSpecies'], axis=1, inplace=True)
X_train = np.asarray(train, dtype=np.float32)

X_train = preprocessing.scale(X_train)

X_test = np.asarray(test, dtype=np.float32)
X_test = preprocessing.scale(X_test)

input_size = len(X_train[0])

learning_rate = theano.shared(np.float32(0.1))

net = NeuralNet(
    layers=[  
        ('input', InputLayer),
        ('hidden1', DenseLayer),
        ('dropout1', DropoutLayer),
        ('hidden2', DenseLayer),
        ('dropout2', DropoutLayer),
        ('output', DenseLayer),
    ],
    # layer parameters:
    input_shape=(None, input_size), 

    hidden1_num_units=850, 
    dropout1_p=0.1,

    hidden2_num_units=200, 
    dropout2_p=0.10,
    
    output_nonlinearity=sigmoid, 
    output_num_units=1, 

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=learning_rate,
    update_momentum=0.91,

    # Decay the learning rate
    on_epoch_finished=[
        AdjustVariable(learning_rate, target=0, half_life=4),
    ],

    # This is silly, but we don't want a stratified K-Fold here
    # To compensate we need to pass in the y_tensor_type and the loss.
    regression=True,
    y_tensor_type = T.imatrix,
    objective_loss_function = binary_crossentropy,

    max_epochs = 50, 
    eval_size=0.2,
    verbose=1,
)

X_train, y_train = shuffle(X_train, y_train, random_state=888)

net.fit(X_train, y_train)

y_pred = net.predict_proba(X_test)[:, 0]

submission.WnvPresent = y_pred
submission.to_csv("ourSub_lasagne_cleaned_no_count_850_200.csv", index=False)
