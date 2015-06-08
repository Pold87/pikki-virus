import os
import pandas as pd
import numpy as np

foldername = os.getcwd() + '/submissions'

filenames = ['submission_1.csv', 'submission_2.csv', 
             'submission_3.csv', 'submission_4.csv', 
             'submission_5.csv']

by_folder = 1   # 1 := take all submissions from 'foldername'
                # 2 := take all submission listed in 'filenames'  

comb = 0
if by_folder == 1:
    cnt = 0
    for subdir, dirs, files in os.walk(foldername):
        for file in files:
            df = np.genfromtxt(os.path.join(subdir, file), delimiter=',', skip_header=1)
            comb = comb + df[:,1]
            cnt = cnt + 1
    comb = comb / cnt
else:
    for file in filenames:
        df = np.genfromtxt(file, delimiter=',', skip_header=1)
        comb = comb + df[:,1]
    comb = comb / len(filenames)

subm_df = pd.read_csv(filenames[0])
subm_df['WnvPresent'] = comb.astype(float)
subm_df.to_csv('bag_subm_mean.csv', index=False, float_format='%.4f')