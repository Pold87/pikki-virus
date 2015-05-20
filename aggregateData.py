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
df_train = pd.read_csv('train.csv')

"""
read in weather data
"""
df_weather = pd.read_csv('weather.csv')

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
    long_dist = long2 - long1
    lat_dist = lat2 - lat1
    a = (sin(lat_dist/2))**2 + cos(lat1) * cos(lat2) * (sin(long_dist/2))**2
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
    return dt.datetime.strptime(str, '%m/%d/%Y').date()

df_train.Date = df_train.Date.map(str_to_date) 
df_weather.Date = df_weather.Date.map(str_to_date)    

"""
for each trap location, calculate which weather station is closer and store in a dataframe.
If difference between distances is smaller than 5km, assign 0. This will later indicate that
the average of the measurements should be used.
"""
lat_station1= 41.995
lon_station1= -87.933 
lat_station2= 41.786
lon_station2= -87.752

def closer_ws(trap_location):
    lat_trap = trap_location['latitude']
    lon_trap = trap_location['longitude']
    dist1 = haversine(lat_trap, lon_trap, lat_station1, lon_station1)
    dist2 = haversine(lat_trap, lon_trap, lat_station2, lon_station2)
    #if difference between distances to weather stations is less than 5km, assign 0
    if np.abs(dist1-dist2) < 5:
        ws_id = 0
    #if distance to ws1 is smaller, ws1 is closer ws and vice versa
    elif dist1 < dist2:
        ws_id = 1
    else: 
        ws_id=2
    return ws_id
        

df_trap_loc['closer_station']= df_trap_loc.apply(closer_ws, axis=1)

"""
calculate average weekly weather variable (temp/precipitation), with Jan 1st 2007 being starting day of
first week. This is calculated separately for the two weather stations
TODO: not separated by years yet. Should be done, since as it is now, some days from
previous fall may be averaged together with days in May
"""
def weekly_avrg(df,new_col_name,var_col_name):
    #add new column for weekly temp average
    df[new_col_name]=pd.Series(index=df.index)
    #group average daily temp column by weather station
    for station_id, station in df.groupby(['Station'])[var_col_name]:
       #chunk this into 7-day chunks 
       for ind, week in station.groupby(np.arange(len(station))//7):
            #calculate average temp for 7 days
            week_avrg = week.mean()
            #save in column Tavg_week
            for index in week.index:
                df.loc[index,new_col_name]=week_avrg
            
weekly_avrg(df_weather,'Tavg_week','Tavg')
weekly_avrg(df_weather,'precip_week','Precipitation')
            
"""
add two more columns to weather data that say whether it was a heat week or a cool
week (similar to heat/cool columns that are already in the data for single days).
Since it was found optimal in the paper by Ruiz, here 22 degrees Celcius (71.6 degrees Fahrenheit)
are used.
"""
def heatcoolhelper(temp):
    return temp - 71.6
#add new columns for Heat Degree Week and Cool Degree Week
df_weather['heat_dw'] = df_weather.Tavg_week.apply(heatcoolhelper)
df_weather['cool_dw'] = df_weather.heat_dw
#for heat week keep temperature differences above Tbase
df_weather.heat_dw[df_weather.heat_dw < 0] = 0
#for cool week keep those below Tbase
df_weather.cool_dw[df_weather.cool_dw > 0] = 0
df_weather.cool_dw = df_weather.cool_dw.apply(np.abs)

"""
add a column to training data that says whether measurement day was in a cooling degree
week or heating degree week at closer weather station
TODO: check if this cannot be made more efficient by grouping by date and trap, because
currently the same value is assigned to many rows, and it is probably faster to assign all rows
with same value at once
"""
def heatcool(row,weather_var):
    #Extract which WS is closer
    closer= df_trap_loc.loc[row.Trap,'closer_station']
    #How much of a heat/cool week was it on measurement day
    week_ws1= df_weather[weather_var][(df_weather.Date == row.Date) & (df_weather.Station==1)]
    week_ws2= df_weather[weather_var][(df_weather.Date == row.Date) & (df_weather.Station==2)]
    #if both weather station are approximately same distance away, average data together
    if closer==0:
        week= (week_ws1.item() + week_ws2.item())/2
    #if one station is closer than the other, take the measurement of that station
    elif closer==1:
        week= week_ws1.item()
    else:
        week= week_ws2.item()
    return week
    
df_train['heat_week']=df_train.apply(heatcool, axis=1, args= ('heat_dw',))
df_train['cool_week']=df_train.apply(heatcool, axis=1, args= ('cool_dw',))
df_train['precip_week']=df_train.apply(heatcool, axis=1, args=('precip_week',))




        
        
        
        

            