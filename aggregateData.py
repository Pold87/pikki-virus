import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import datetime as dt


def haversine(lat1,long1,lat2,long2):

    """
    haversine distance (in km) between two coordinates in lat and long
    """

    earth_radius = 6371 #in kilometers

    lat1 = radians(lat1)
    long1 = radians(long1)
    lat2 = radians(lat2)
    long2 = radians(long2)

    long_dist = long2 - long1
    lat_dist = lat2 - lat1

    a = (sin(lat_dist / 2)) ** 2 + cos(lat1) * cos(lat2) * (sin(long_dist / 2)) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = earth_radius * c

    return distance

def closer_ws(trap_location):

    """
    for each trap location, calculate which weather station is closer and store in a dataframe df_trap_loc.
    If difference between distances is smaller than 5km, assign 0. This will later indicate that
    the average of the measurements should be used.
    """

    # weather station coordinates
    lat_station1 = 41.995
    lon_station1 = -87.933 
    lat_station2 = 41.786
    lon_station2 = -87.752
    
    lat_trap = trap_location['latitude']
    lon_trap = trap_location['longitude']
    dist1 = haversine(lat_trap, lon_trap, lat_station1, lon_station1)
    dist2 = haversine(lat_trap, lon_trap, lat_station2, lon_station2)
    
    # if difference between distances to weather stations is less than 5km, assign 0
    if np.abs(dist1-dist2) < 5:
        ws_id = 0
    # if distance to ws1 is smaller, ws1 is closer ws and vice versa
    elif dist1 < dist2:
        ws_id = 1
    else: 
        ws_id = 2

    return ws_id


def create_trap_df(df):

    # what are the unique traps of measurements?
    trap_series = df.loc[:, 'Trap']
    unique_traps = trap_series.unique()
    
    # df that stores latitude and longitude for each trap id, with trap ids as indices
    df_trap_loc = pd.DataFrame(index=unique_traps, columns=['latitude','longitude'])

    for name, group in df.groupby('Trap'):
        group.reset_index(inplace=True)   
        df_trap_loc.ix[name,:] = [group.ix[0,'Latitude'], group.ix[0,'Longitude']]

    df_trap_loc['closer_station']= df_trap_loc.apply(closer_ws, axis=1)

    return df_trap_loc


def calc_trap_distance_matrix(df):

    """
    compute distance between the different traps and store them in a matrix
    """

    df_trap_loc = create_trap_df(df)
    
    #create trap x trap matrix
    trap_distance_matrix = pd.DataFrame(index=df_trap_loc.index, columns=df_trap_loc.index)

    for trap1 in unique_traps:

        #ensures that symmetrical values in matrix are only computed once
        k = len(unique_traps)-1 
        unique_traps_shorter = np.delete(unique_traps, k, axis=0) 
        k = k-1

        #get lat and long of first trap
        lat_trap1 = df_trap_loc.loc[trap1,'latitude']
        long_trap1 = df_trap_loc.loc[trap1,'longitude']

        for trap2 in unique_traps_shorter:

            #get lat and long of second trap
            lat_trap2 = df_trap_loc.loc[trap2,'latitude']
            long_trap2 = df_trap_loc.loc[trap2,'longitude']

            #compute the haversine distance between the two traps and fill in twice in matrix
            dist= haversine(lat_trap1,long_trap1,lat_trap2,long_trap2)
            trap_distance_matrix.loc[trap1,trap2]=dist
            trap_distance_matrix.loc[trap2,trap1]=dist

    return trap_distance_matrix
    

def find_closest_per_trap(trap_distances, n):

    """
    find n closest traps to each trap
    returns boolean mask of closest traps
    """
    
    #sort values by distance
    sorted_dists = trap_distances.order()

    #choose closest n
    top_n = sorted_dists.iloc[:n]
    
    #get trap ids of closest n
    index_array=top_n.index.values
    
    #boolean mask of original trap_distance series, with n closest = TRUE
    trap_distances = trap_distances.index.isin(index_array)
    
    return trap_distances

  
def str_to_date(str):

    """
    convert dates into datetime.date objects so that differences in days can be determined
    """
    
    return dt.datetime.strptime(str, '%Y-%m-%d').date()


def extract_year(date):

    """
    add additional column "year" to weather dataframe in order to group the data by year
    """
    
    return date.year        

def weekly_avrg(df, new_col_name, var_col_name):

    """
    Calculate average weekly weather variable (temp/precipitation),
    with May 1st 2007 being starting day of first week. This is
    calculated separately for the two weather stations

    """
    
    #add new column for weekly temp average
    df[new_col_name] = pd.Series(index=df.index)
    
    #group average daily temp column by weather station
    for station_id, station in df.groupby(['Station', 'Year'])[var_col_name]:

        #chunk this into 7-day chunks 
        for ind, week in station.groupby(np.arange(len(station)) // 7):

            #calculate average temp for 7 days
            week_avrg = week.mean()
            
            #save in column Tavg_week
            for index in week.index:
                df.loc[index,new_col_name] = week_avrg

    return df
                        
def heat_degree_week(temp):

    """
    Add two more columns to weather data that say whether it was a
    heat week or a cool week (similar to heat/cool columns that are
    already in the data for single days).  Since it was found optimal
    in the paper by Ruiz, here 22 degrees Celcius (71.6 degrees
    Fahrenheit) are used.
    """
    
    return max(temp - 71.6, 0)

def cool_degree_week(temp):

    return max(71.6 - temp, 0) 


def mov_window(chunk, x, w_var):

    """
    In the paper by ruiz a 3-week and a 5-week moving window are calculated over the
    precipitation. This method calculates an x-week moving window for the average of
    a specific weather variable w_var.
    """
    
    #iterate over each group by  
    chunk[str(x) + "_week_avrg" + w_var] = pd.rolling_mean(chunk[w_var], window=7 * x, min_periods=1)

    return chunk


def add_weather_var(row, df_weather, df_trap_loc, weather_var):
 
    """
    Add a column to training data that says whether measurement day
    was in a cooling degree week or heating degree week at closer
    weather station TODO: check if this cannot be made more efficient
    by grouping by date and trap, because currently the same value is
    assigned to many rows, and it is probably faster to assign all
    rows that get same value at once.
    """
    
    #Extract which WS is closer
    
    closer = df_trap_loc.loc[row.Trap,'closer_station']
    
    #What was the weather variable like on day of interest
    weather_var_ws1 = df_weather[weather_var][(df_weather.Date == row.Date) & (df_weather.Station == 1)]
    weather_var_ws2 = df_weather[weather_var][(df_weather.Date == row.Date) & (df_weather.Station == 2)]

    #if both weather station are approximately same distance away, average data together
    if closer == 0:
        avg = (weather_var_ws1.item() + weather_var_ws2.item()) / 2
    #if one station is closer than the other, take the measurement of that station
    elif closer == 1:
        avg = weather_var_ws1.item()
    else:
        avg = weather_var_ws2.item()

    return avg
    

def load_weather(path='weather.csv'):

    df_weather = pd.read_csv(path)

    df_weather.Tavg = df_weather.Tavg.replace('M', np.nan, regex=True)
    df_weather.PrecipTotal = df_weather.PrecipTotal.replace('T', 0.005, regex=True)
    df_weather.PrecipTotal = df_weather.PrecipTotal.replace('M', np.nan, regex=True)

    df_weather.Tavg = df_weather.Tavg.astype(float)
    df_weather.PrecipTotal = df_weather.PrecipTotal.astype(float)

    df_weather.Date = df_weather.Date.map(str_to_date)
    df_weather['Year'] = df_weather.Date.apply(extract_year)

    # Add weekly average
    df_weather = weekly_avrg(df_weather,'Tavg_week','Tavg')
    df_weather = weekly_avrg(df_weather,'precip_week','PrecipTotal')
    
    #add new columns for Heat Degree Week and Cool Degree Week
    df_weather['heat_dw'] = df_weather.Tavg_week.map(heat_degree_week)
    df_weather['cool_dw'] = df_weather.Tavg_week.map(cool_degree_week)

    df_weather = df_weather.groupby(['Station','Year']).apply(mov_window, 3, 'PrecipTotal')
    df_weather = df_weather.groupby(['Station','Year']).apply(mov_window, 5, 'PrecipTotal')

    return df_weather


def load_train(path='train.csv'):

    # read in weather data
    df_weather = load_weather()

    df_train = pd.read_csv(path)
    df_train.Date = df_train.Date.map(str_to_date)

    df_trap_loc = create_trap_df(df_train)
    
    df_train['heat_week'] = df_train.apply(add_weather_var,
                                           axis=1,
                                           args=(df_weather, df_trap_loc, 'heat_dw',))
    
    df_train['cool_week'] = df_train.apply(add_weather_var,
                                           axis=1,
                                           args=(df_weather, df_trap_loc, 'cool_dw',))
    
    df_train['precip_week'] = df_train.apply(add_weather_var,
                                             axis=1,
                                             args=(df_weather, df_trap_loc, 'precip_week',))
    
    df_train['3_week_avrgPrecipTotal'] = df_train.apply(add_weather_var,
                                              axis=1,
                                              args=(df_weather, df_trap_loc, '3_week_avrgPrecipTotal',))
    
    df_train['5_week_avrgPrecipTotal'] = df_train.apply(add_weather_var,
                                                        axis=1,
                                                        args=(df_weather, df_trap_loc, '5_week_avrgPrecipTotal',))

    return df_train


def load_test(path='test.csv'):

    # read in weather data
    df_weather = load_weather()

    df_test = pd.read_csv(path)
    df_test.Date = df_test.Date.map(str_to_date)

    df_trap_loc = create_trap_df(df_test)
                                                                                                          
    df_test['heat_week'] = df_test.apply(add_weather_var,                                               
                                           axis=1,                                                        
                                           args=(df_weather, df_trap_loc, 'heat_dw',))                    
                                                                                                          
    df_test['cool_week'] = df_test.apply(add_weather_var,                                               
                                           axis=1,                                                        
                                           args=(df_weather, df_trap_loc, 'cool_dw',))                    
    
    df_test['precip_week'] = df_test.apply(add_weather_var,                                             
                                           axis=1,                                                      
                                           args=(df_weather, df_trap_loc, 'precip_week',))              
    
    df_test['3_week_avrgPrecipTotal'] = df_test.apply(add_weather_var,                                          
                                            axis=1,                                                     
                                            args=(df_weather, df_trap_loc, '3_week_avrgPrecipTotal',))
                                                                                                          
    df_test['5_week_avrgPrecipTotal'] = df_test.apply(add_weather_var,                                          
                                            axis=1,                                                     
                                            args=(df_weather, df_trap_loc, '5_week_avrgPrecipTotal',))


    return df_test


if __name__ == '__main__':

    print("Starting")
    # read in training data
    df_train = load_train()
    df_train.to_csv("train_filled.csv", index=False)

    print("Reading Test")
    # read in test data
    df_test = load_test()
    df_test.to_csv("test_filled.csv", index=False)

    print("Finished")
    
    # Create distance matrices
    #trap_distance_matrix_train = calc_trap_distance_matrix(df_train)
    #trap_distance_matrix_test = calc_trap_distance_matrix(df_test)

    # Find ten nearest neigbors per trap
    #closest_ten_traps_train = trap_distance_matrix_train.apply(find_closest_per_trap, axis=0, args = (10,))
    #closest_ten_traps_test = trap_distance_matrix_test.apply(find_closest_per_trap, axis=0, args = (10,))





    




        
        
        
        

            
