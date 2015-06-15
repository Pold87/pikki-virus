import pandas as pd

# Load train and test set
df_train = pd.read_csv('TRAIN/train.csv')
df_test = pd.read_csv('TEST/test.csv')

# Create series of trap names, latitude and longitude values
trap_series = df_train.loc[:, 'Trap'].append(df_test.loc[:, 'Trap'])
lat_series = df_train.loc[:,'Latitude'].append(df_test.loc[:, 'Latitude'])
long_series = df_train.loc[:,'Longitude'].append(df_test.loc[:, 'Longitude'])

# Find unique trap names
unique_traps = trap_series.unique()

# Find trap names that correspond to more than one location and assign each
# a new name
# e.g. 'T009' --->  'T009_0'
#                   'T009_1'                   
for trap in unique_traps:
    lat_ser = df_train[df_train.Trap==trap].Latitude.append(df_test[df_test.Trap==trap].Latitude)
    long_ser = df_train[df_train.Trap==trap].Longitude.append(df_test[df_test.Trap==trap].Longitude)
    if len(lat_ser.unique()) > 1 or len(long_ser.unique()) > 1:
        
        coor_list = {}
        for i in lat_ser.index:
            try:
                if (lat_ser.loc[i], long_ser.loc[i]) not in coor_list.keys():
                    coor_list[(lat_ser.loc[i], long_ser.loc[i])]=trap+'_{}'.format(len(coor_list))
            except: 
                pass
        for ind in coor_list:
            mask_train = (df_train['Trap']==trap) & (df_train['Latitude']==ind[0]) & (df_train['Longitude']==ind[1])
            mask_test = (df_test['Trap']==trap) & (df_test['Latitude']==ind[0]) & (df_test['Longitude']==ind[1])
            df_train.ix[mask_train, 'Trap'] = coor_list[ind]     
            df_test.ix[mask_test, 'Trap'] = coor_list[ind]
            
# Save corrected train and test sets
df_train.to_csv("TRAIN/train_unified_traps.csv", index=False)
df_test.to_csv("TEST/test_unified_traps.csv", index=False)
