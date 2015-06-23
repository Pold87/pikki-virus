import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn import cross_validation, preprocessing, metrics


# Read in combined dfs
df_train = pd.read_csv("subset_A_train.csv")
df_test = pd.read_csv("subset_A_test.csv")

#read in support

train_support = pd.read_csv('train.csv')
test_support = pd.read_csv('test.csv')

# Assign unspecified culex to pipiens / restuans (only in test set)
df_test.Species[df_test.Species == 'UNSPECIFIED CULEX'] = 'CULEX PIPIENS/RESTUANS'


#take care that train and test have the same feature space (empty dummies don't help us)
droplist1 = [var for var in df_train if var not in df_test]
droplist2 = [var for var in df_test if var not in df_test]
df_train = df_train.drop(droplist1,axis=1)
df_test = df_test.drop(droplist2,axis=1)



#get nummosquitos back
df_train['NumMosquitos'] = train_support['NumMosquitos']
#print('\n\n\n\n\n')
#print(df_train.NumMosquitos.max())
#print('\n\n\n\n\n')

# make counter for counting. Useless comment is useless.
df_train['counter'] = 1


# Create copy of train df and drop string features
X = df_train.copy()

print('\n\n\n\n\n')
print(df_train.NumMosquitos.max())
#print(X.NumMosquitos.max())
print('\n\n\n\n\n')

string_features = ['Address','Street','AddressNumberAndStreet','CodeSum']

#create dumps in case we want to use the string_features ever again. (spoiler: we actually don't)
X_dump = X[string_features]
df_dump = df_test[string_features]
X = X.drop(string_features,axis=1)
df_test = df_test.drop(string_features,axis=1)
df_train = df_train.drop(string_features,axis=1)

#group
groups = ['Date','Trap','Species']
X_tmp = X.join(X.groupby(groups)['NumMosquitos'].sum(), on= groups,rsuffix='_counter')
y = X_tmp.NumMosquitos_counter

#make dummies
X_tmp = pd.get_dummies(X_tmp, columns=['Species','Trap'])
df_test = pd.get_dummies(df_test, columns=['Species','Trap'])

#get rid of Date
X_tmp = X_tmp.drop(['Date'],axis=1)
df_test = df_test.drop(['Date'],axis=1)
###regress###

#clf = GradientBoostingRegressor(n_estimators = 4000,max_depth=15,learning_rate = 0.01)
clf = GradientBoostingRegressor(n_estimators = 4,max_depth=15,learning_rate = 0.01)
random_state= 46

#X = scale(X_tmp)
df_test = df_test.drop([var for var in df_test.columns if var not in X_tmp.columns],axis=1)
X_tmp =X_tmp.drop([var for var in X_tmp.columns if var not in df_test.columns],axis=1)
clf.fit(X_tmp,y)
yhat = clf.predict(df_test)


#set variables
df_train['count_mosquitos'] = y
df_test['count_mosquitos'] = yhat

#restore old vars


df_train['WnvPresent'] = train_support['WnvPresent']
df_train.Date = train_support.Date
df_test.Date = test_support.Date



#write to file

df_test.to_csv('subset_C_test.csv',index_label='Id')
df_train.to_csv('subset_C_train.csv',index_label='Id')



