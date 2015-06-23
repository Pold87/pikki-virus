import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn import cross_validation, preprocessing, metrics


# Read in combined dfs
df_train = pd.read_csv("subset_A_train.csv")
df_test = pd.read_csv("subset_A_test.csv")


# Assign unspecified culex to pipiens / restuans
df_test.Species[df_test.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'

# Create copy of train df and drop string features
string_features = ['Address','Street','AddressNumberAndStreet','CodeSum']

X = df_train.copy()
X = X.drop(string_features,axis=1)

# Create groups for multiple rows which have the same value

groups = ['Date','Trap','Species']
tmp = X.groupby(groups)['NumMosquitos'].sum()
X_tmp = X.join(X.groupby(groups)['NumMosquitos'].sum(), on= groups,rsuffix='_counter')

# New WnvPresent
y = X_tmp.NumMosquitos_counter

X_tmp = X_tmp.drop(['Trap',
                    'Date',
                    'NumMosquitos',
                    'NumMosquitos_counter',
                    'WnvPresent',
                    'counter',
                    'Unnamed: 0'])


# Species to labels
# TODO dummy variables might be better!!

X_tmp.Species = species_encoder.fit_transform(X.Species)
df_test.Species = species_encoder.transform(df_test.Species)
df_train.species = species_encoder.transform(df_train.Species)

df_train['count_mosquitos'] = y

# Add conditional features
for feature in df_train.columns[:-1]:
    featureprior = df_train.groupby(
        [feature])['WnvPresent'].sum() /df_train.groupby([feature])['WnvPresent'].count()
    df_train = df_train.join(featureprior,
                             on=feature,
                             rsuffix='_conditional_' + feature)
    df_test = df_test.join(featureprior,
                           on=feature,
                           rsuffix = '_conditional_' + feature)


df_test = df_test.dropna(axis=1)#.shape    

not_droplist = [var for var in df_test.columns]
not_droplist.append('WnvPresent')

droplist = [var for var in df_train.columns if var not in not_droplist]
df_train = df_train.drop(droplist,axis=1)

df_train.to_csv('subset_B_train.csv')
df_test.to_csv('subset_B_test.csv')
