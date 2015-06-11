from __future__ import division

import pandas as pd
import numpy as np

# Read in combined dfs
df_train = pd.read_csv("train_filled_new.csv")
df_test = pd.read_csv("test_filled_new.csv")

groups = ['Date','Trap','Species']

X = df_train.join(
    df_train.groupby(groups)['WnvPresent'].sum(),
    on=groups,
    rsuffix='_grouped')

# Create mask ("In which rows is Wnv Present?")
msk = X.WnvPresent_grouped > 0

X['WnvPresent'][msk] = 1
X['MIR'] = X['WnvPresent_grouped'] / X['NumMosquitos']

X = X.drop(['WnvPresent_grouped'], axis=1)
X = X.drop_duplicates()

# Save to file 
X.to_csv('volker_unique_train.csv', index_label='Id')
