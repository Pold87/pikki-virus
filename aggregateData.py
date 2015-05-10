from os import listdir, path
#from pandas import io
import pandas as pd
import numpy as np
#import re
#from scipy import stats
from math import sin, cos, sqrt, atan2, radians
import datetime as dt

"""
read in training data
"""
data_path = 'c:/Users/Franziska/Documents/GitHub/pikki-virus'
files = listdir(data_path)
df_train = pd.read_csv(path.join(data_path, 'train.csv'))
#train_header = list(df_train.columns.values)

"""
what are the unique dates of measurements?
"""
date_series = df_train.loc[:,'Date']
unique_dates = date_series.unique()

"""
what are the unique traps of measurements?
"""
trap_series = df_train.loc[:, 'Trap']
unique_traps = trap_series.unique()

"""
haversine distance (in km) between two coordinates in lat and long
"""
def haversine(lat1,long1,lat2,long2):
    earth_radius=6371 #in kilometers
    lat1=radians(lat1)
    long1=radians(long1)
    lat2=radians(lat2)
    long2=radians(long2)
    long_dist = long_j - long_i
    lat_dist = lat_j - lat_i
    a = (sin(lat_dist/2))**2 + cos(lat_i) * cos(lat_j) * (sin(long_dist/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = earth_radius * c
    return distance

"""
compute distance between the different traps and store them in a matrix
"""
#create trap x trap matrix
trap_distance_matrix = pd.DataFrame(index=unique_traps, columns=unique_traps)
#df that stores latitude and longitude for each trap id, with trap ids as indices
df_trap_loc = pd.DataFrame(index=unique_traps, columns=['latitude','longitude'])
for name, group in df_train.groupby('Trap'):
    group.reset_index(inplace=True)   
    df_trap_loc.ix[name,:] = [group.ix[0,'Latitude'], group.ix[0,'Longitude']]

for i in unique_traps:
    #ensures that symmetrical values in matrix are only computed once
    k=len(unique_traps)-1 
    unique_traps_shorter = np.delete(unique_traps, k, axis=0) 
    k=k-1
    #get lat and long of first trap
    lat_i = df_trap_loc.loc[i,'latitude']
    long_i = df_trap_loc.loc[i,'longitude']
    for j in unique_traps_shorter:
        #get lat and long of second trap
        lat_j = df_trap_loc.loc[j,'latitude']
        long_j = df_trap_loc.loc[j,'longitude']
        #compute the haversine distance between the two traps and fill in twice in matrix
        dist= haversine(lat_i,long_i,lat_j,long_j)
        trap_distance_matrix.loc[i,j]=dist
        trap_distance_matrix.loc[j,i]=dist
    
"""
find n closest traps to each trap
returns boolean mask of closest traps
"""
def find_closest_per_trap(trap_distances,n):
    #sort values by distance
    sorted_dists = trap_distances.order()
    #choose closest n
    top_n = sorted_dists.iloc[:n]
    #get trap ids of closest n
    index_array=top_n.index.values
    #boolean mask of original trap_distance series, with n closest = TRUE
    trap_distances = trap_distances.index.isin(index_array)
    return trap_distances

  
closest_ten_traps = trap_distance_matrix.apply(find_closest_per_trap, axis=0, args = (10,))


"""
convert dates into datetime.date objects so that differences in days can be determined
"""
def str_to_date(str):
    return dt.datetime.strptime(str, '%Y-%m-%d').date()

df_train.Date = df_train.Date.map(str_to_date)     