
# coding: utf-8

# In[39]:

import pandas as pd


# In[40]:

train = pd.read_csv('../No_Species_Dataset/train_without_Species.csv')


# In[41]:

test = pd.read_csv('../No_Species_Dataset/test_agg_without.csv')


# In[42]:

'Species' in train.columns


# In[43]:

y = train.WnvPresent


# In[44]:

X = train.copy()


# In[45]:

X = X.drop('WnvPresent',axis=1)


# In[46]:

strin = ['CodeSum',  'AddressNumberAndStreet' ,'Address', 'Street','Date']


# In[47]:

X = X.drop(strin, axis=1)


# In[48]:

X.dtypes[:-5]


# In[ ]:

from sklearn.cross_validation import train_test_split, ShuffleSplit,cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


# In[50]:

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3)


# In[51]:

len(X)


# In[52]:

cv = ShuffleSplit(len(X))


# In[56]:

clf = GradientBoostingClassifier()
clf = RandomForestClassifier()


# In[ ]:

cross_val_score(clf,X,y,cv=cv,n_jobs=-1)


# In[ ]:



