import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import datetime as dt
from sklearn import preprocessing
from scipy.spatial import distance

def closer_ws(trap_location):

    """
    For each trap location, calculate which weather station is closer and store in a dataframe df_trap_loc.
    If difference between distances is smaller than 5km, assign 0. This will later indicate that
    the average of the measurements should be used.
    """

    y_station1 = 55.041
    x_station1 = 5.580
    y_station2 = 31.802
    x_station2 = 20.653
    
    y_trap = trap_location['YCoor']
    x_trap = trap_location['XCoor']
    dist1 = distance.euclidean((y_station1, x_station1), (y_trap, x_trap))
    dist2 = distance.euclidean((y_station2, x_station2), (y_trap, x_trap))
    
    if dist1 < dist2:
        ws_id = 1
    else: 
        ws_id = 2

    return ws_id

  
def str_to_date(str):

    """
    convert dates into datetime.date objects so that differences in days can be determined
    """
    
    return dt.datetime.strptime(str, '%Y-%m-%d').date()
    
def date_to_calweek(date_obj):
    
    """
    determine which calendar week of the year a specific date was on
    """
    
    return dt.date.isocalendar(date_obj)[1]

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
    
def average_shifted(chunk, x, w_var):
    
    """
    Shifts the moving average column for the 1 week downward by a x weeks.
    Consequently calculates the average temp/precipitation x weeks ago. 
    This naturally results in several NaN values at the beginning of each year and thus
    should only be done for a small number of weeks (below, I selected 1 to 4)
    """
    
    col = '1_week_avrg' + w_var
    
    tmp = chunk[col].shift(x*7)
    
    tmp[tmp.index[tmp.apply(np.isnan)]]=0   
    
    #chunk[w_var + '_' + str(x) + '_weeks_ago'] = chunk[col].shift(x*7)
    chunk[w_var + '_' + str(x) + '_weeks_ago'] = tmp
    
    #print('chunk', chunk)
    
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
    
def load_weather(path='WEATHER/weather_hand.csv'):

    """
    Read in weather.csv and pre-process it
    """

    df_weather = pd.read_csv(path)

    # Handle missing values in df
    df_weather = df_weather.replace('T$', 0.005, regex=True)
    df_weather = df_weather.replace('M$', np.nan, regex=True)
    df_weather = df_weather.replace('-$', np.nan, regex=True)

    # Drop Water1 column (is always NaN)
    df_weather = df_weather.drop(['Water1'], axis=1)
    
    # Impute missing values with mean

    # Depart is only available from one weather station !
    imp_depart = preprocessing.Imputer(axis=1, strategy='mean')
    depart = imp_depart.fit_transform(df_weather.Depart)
    df_weather.Depart = imp_depart.fit_transform(df_weather.Depart)[0]
    
    imp_wetbulb = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.WetBulb = imp_wetbulb.fit_transform(df_weather.WetBulb)[0]

    imp_sunrise = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.Sunrise = imp_sunrise.fit_transform(df_weather.Sunrise)[0]
    
    imp_sunset = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.Sunset = imp_sunset.fit_transform(df_weather.Sunset)[0]
    
    imp_depth = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.Depth = imp_depth.fit_transform(df_weather.Depth)[0]

    imp_snowfall = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.SnowFall = imp_snowfall.fit_transform(df_weather.SnowFall)[0]
    
    imp_preciptotal = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.PrecipTotal = imp_preciptotal.fit_transform(df_weather.PrecipTotal)[0]
    
    imp_stnpressure = preprocessing.Imputer(axis=1, strategy='mean')
    df_weather.StnPressure = imp_stnpressure.fit_transform(df_weather.StnPressure)[0]
    
    # Change type of column values
    df_weather.Tavg = df_weather.Tavg.astype(float)
    df_weather.PrecipTotal = df_weather.PrecipTotal.astype(float)

    df_weather.Date = df_weather.Date.map(str_to_date)
    df_weather['Year'] = df_weather.Date.apply(lambda x: x.year)
    df_weather['Month'] = df_weather.Date.apply(lambda x: x.month)


    # Add weekly average of temperature and and precipitation		
    df_weather = weekly_avrg(df_weather,'Tavg_week','Tavg')		
    df_weather = weekly_avrg(df_weather,'precip_week','PrecipTotal')
    
    # Add new columns for Heat Degree Week and Cool Degree Week
    df_weather['heat_dw'] = df_weather.Tavg_week.map(heat_degree_week)
    df_weather['cool_dw'] = df_weather.Tavg_week.map(cool_degree_week)

    # TODO!
    #df_weather['heat_dw_shifted2'] = df_weather.heat_dw
    #df_weather['cool_dw_shifted2'] = df_weather.cool_dw

        
    # Add moving windows
    for weeks in range(1, 24):

        # Add moving window for precipitation
        df_weather = df_weather.groupby(['Station','Year']).apply(mov_window, weeks, 'PrecipTotal')

        # Add moving moving window for temperature
        df_weather = df_weather.groupby(['Station','Year']).apply(mov_window, weeks, 'Tavg')

    # Add shifted moving windows, shifted by 1 to 4 weeks 
    for weeks in range(1, 4):
        
        # Add shifted moving window for precipiation
        df_weather = df_weather.groupby(['Station','Year']).apply(average_shifted, weeks, 'PrecipTotal')
        
        #Add shifted moving window for temperature
        df_weather = df_weather.groupby(['Station','Year']).apply(average_shifted, weeks, 'Tavg')        

    
    return df_weather

def load_data(df_weather, path='train.csv'):

    """
    Read in train.csv or test.csv and preprocess the data frame
    """

    df = pd.read_csv(path)
    df.Date = df.Date.map(str_to_date)

    # For each row, determine which weather station is closer
    df['Station'] = df.apply(closer_ws, axis=1)
    
    # Merge train/test data frame and weather data 
    merged_df = pd.merge(df, df_weather, on=['Station', 'Date'])

    return merged_df

if __name__ == '__main__':

    print("Starting")
    print("Reading in weather data")
    # read in weather data
    df_weather = load_weather()
    print("Processing Training Data")
    # read in training data
    df_train = load_data(df_weather, "TRAIN/train_hex.csv")
    df_train.to_csv("TRAIN/train_hex_weather.csv", index=False)
    
    print("Processing Complementary Data")
    # read in training data
    df_comp = load_data(df_weather, "COMP/comp_hex.csv")
    df_comp.to_csv("COMP/comp_hex_weather.csv", index=False)

    print("Processing Test Data")
    # # read in test data
    df_test = load_data(df_weather, "TEST/test_hex.csv")
    df_test.to_csv("TEST/test_hex_weather.csv", index=False)

    
    print("Finished")