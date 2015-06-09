import pandas as pd

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
trap_series = df_train.loc[:, 'Trap'].append(df_test.loc[:, 'Trap'])
lat_series = df_train.loc[:,'Latitude'].append(df_test.loc[:, 'Latitude'])
long_series = df_train.loc[:,'Longitude'].append(df_test.loc[:, 'Longitude'])
unique_traps = trap_series.unique()

for el in unique_traps:
    lat_ser = df_train[df_train.Trap==el].Latitude.append(df_test[df_test.Trap==el].Latitude)
    long_ser = df_train[df_train.Trap==el].Longitude.append(df_test[df_test.Trap==el].Longitude)
    if len(lat_ser.unique()) > 1 or len(long_ser.unique()) > 1:
        
        coor_list = {}
        for i in lat_ser.index:
            try:
                if (lat_ser.loc[i], long_ser.loc[i]) not in coor_list.keys():
                    coor_list[(lat_ser.loc[i], long_ser.loc[i])]=el+'_{}'.format(len(coor_list))
            except: 
                pass
        for ind in coor_list:
            mask_train = (df_train['Trap']==el) & (df_train['Latitude']==ind[0]) & (df_train['Longitude']==ind[1])
            mask_test = (df_test['Trap']==el) & (df_test['Latitude']==ind[0]) & (df_test['Longitude']==ind[1])
            df_train.ix[mask_train, 'Trap'] = coor_list[ind]     
            df_test.ix[mask_test, 'Trap'] = coor_list[ind]
            
df_train.to_csv("train_trap_correct.csv", index=False)
df_test.to_csv("test_trap_correct.csv", index=False)
