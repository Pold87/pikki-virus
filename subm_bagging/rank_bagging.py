import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata

foldername = os.getcwd() + '/submissions'

filenames = ['submission_1.csv', 'submission_2.csv', 
             'submission_3.csv', 'submission_4.csv', 
             'submission_5.csv']

by_folder = 1   # 1 := take all submissions from 'foldername'
                # 2 := take all submission listed in 'filenames'  

rank = 0
if by_folder == 1:
    for subdir, dirs, files in os.walk(foldername):
        for file in files:
            df =  pd.read_csv(os.path.join(subdir, file))
            rank = rank + rankdata(df.WnvPresent, method='ordinal')
else:
    for file in filenames:
        df =  pd.read_csv(file)
        rank = rank + rankdata(df.WnvPresent, method='ordinal')

rank = rank / (max(rank) + 1.0)

df = df.copy()
df['WnvPresent'] = rank
df.to_csv('bag_subm_rank.csv', index = False)