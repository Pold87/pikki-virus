import pandas as pd
import numpy as np
from math import sin, cos, sqrt, atan2, radians
import datetime as dt
from scipy.spatial import distance
from itertools import repeat

# Define Hex-grid points (0,0) and (end,end) in (latitude,longitude)
#    , and width of 1 cell (in km)
LATI_ZERO = 41.5 
LONG_ZERO = -88.0
LATI_END  = 42.1 
LONG_END  = -87.4
HEX_CELL_SIZE  = 1.0
HEX_CELL_HEIGHT = HEX_CELL_SIZE * 2.0
HEX_CELL_WIDTH = sqrt(3.0) / 2.0 * HEX_CELL_HEIGHT
HEX_CELL_VERT = HEX_CELL_HEIGHT * 3.0 / 4.0
HEX_CELL_HORI = HEX_CELL_WIDTH

# Load train and test datasets
df_train = pd.read_csv('TRAIN/train_unified_traps.csv')
df_test = pd.read_csv('TEST/test_unified_traps.csv')

# Find unique traps
trap_series = df_train.loc[:, 'Trap'].append(df_test.loc[:, 'Trap'])
unique_traps = trap_series.unique()

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
    
def date_to_year(date_obj):
    
    """
    determine which calendar year a specific date was in
    """
    
    return date_obj.year

def ywd_to_date(year, week, weekday):
    """Convert (year, week, isoweekday) tuple to a datetime.date().

    >>> datetime.date(2013, 7, 12).isocalendar()
    (2013, 28, 5)
    >>> ywd_to_date(2013, 28, 5)
    datetime.date(2013, 7, 12)
    """
    first = dt.date(year, 1, 1)
    first_year, _first_week, first_weekday = first.isocalendar()

    if first_year == year:
        week -= 1
        
    date = first + dt.timedelta(days=week*7+weekday-first_weekday)

    return "{:04d}-{:02d}-{:02d}".format(date.year, date.month, date.day)

# Find all dates, map them to calender weeks, find unique calender weeks
date_series = df_train.loc[:, 'Date'].append(df_test.loc[:, 'Date'])
date_series = date_series.map(str_to_date)
cw_series = date_series.map(date_to_calweek)
unique_dates = cw_series.unique()
# Set 1st calender week at 3 weeks before any data point
start_cw = np.min(unique_dates)-3
end_cw = np.max(unique_dates)
cws = range(start_cw,end_cw+1)
years = [2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]

# Find unique species
species_series = df_test.loc[:, 'Species']
unique_species = species_series.unique()

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
computes coordinates based on row and column of hex_grid
"""    
def calc_pos(row, col, latlong=1):
    
    pt_vert = HEX_CELL_HEIGHT * 0.5 + row * HEX_CELL_VERT
    pt_hori = 0.0
    if row % 2 == 0:
        pt_hori = HEX_CELL_WIDTH * (col + 1.0)
    else:
        pt_hori = HEX_CELL_WIDTH * (col + 0.5)    
        
    return pt_vert, pt_hori

"""
computes to which hex_cell a coordinate belongs
"""
def hex_cell(hex_pos_grid,lat1,long1):
    pt_vert = haversine(LATI_ZERO,LONG_ZERO,lat1,LONG_ZERO)
    pt_hori = haversine(LATI_ZERO,LONG_ZERO,LATI_ZERO,long1)
    
    hex_c = []
    dist = np.inf
    for el in hex_pos_grid:
        tmp_dist = distance.euclidean(hex_pos_grid[el], (pt_vert, pt_hori))
        if tmp_dist < dist:
            hex_c = el
            dist = tmp_dist
            
    return hex_c

"""
creates the hex grid, returns:
- hex_dict:      hex_cell --> trap name (real 'Txxx' or imaginary 'Ixxx')
- hex_pos_grid:  hex_cell --> y- and x- coordinate
- trap_dict:     trap name (only real) --> hex_cell
"""
def create_grid():
    
    hex_grid = np.zeros((np.size(unique_traps),2))
    
    hex_pos_grid = {}
    grid_height = np.ceil(haversine(LATI_ZERO,LONG_ZERO,LATI_END,LONG_ZERO) / HEX_CELL_VERT).astype(int)
    grid_width  = np.ceil(haversine(LATI_ZERO,LONG_ZERO,LATI_ZERO,LONG_END) / HEX_CELL_HORI).astype(int)
    for i in range(0, grid_height+1):
        for j in range(0, grid_width+1):
            if i % 2 == 0:
                hex_pos_grid[(i,j)] = (0.5 * HEX_CELL_HEIGHT + i * HEX_CELL_VERT, (j + 1) * HEX_CELL_HORI)
            else:
                hex_pos_grid[(i,j)] = (0.5 * HEX_CELL_HEIGHT + i * HEX_CELL_VERT, (j + 0.5) * HEX_CELL_HORI)

    trap_dict={}
    for i, el in enumerate(unique_traps): 
        ind = df_test[df_test['Trap'] == el].index.tolist()[0]
        lat1 = df_test['Latitude'].loc[ind]
        long1 = df_test['Longitude'].loc[ind]
        hex_c = hex_cell(hex_pos_grid, lat1,long1)
        hex_grid[i]=hex_c  
        trap_dict[el]=hex_c
    
    hex_dict={}
    
    row_min = (np.min(hex_grid[:,0])-2).astype(int)
    row_max = (np.max(hex_grid[:,0])+2).astype(int)
    col_min = (np.min(hex_grid[:,1])-2).astype(int)
    col_max = (np.max(hex_grid[:,1])+2).astype(int)
    
    hex_grid_2 = pd.DataFrame(data=hex_grid, index=unique_traps) 
    
    imag_trap_cnt = 1
    for row in range(row_min, row_max+1):
        for col in range(col_min, col_max+1):
            tmp=list(hex_grid_2[hex_grid_2[0]==row][hex_grid_2[1]==col].index.values)
            if not tmp:
                hex_dict[(row,col)]=['I{0:04}'.format(imag_trap_cnt)]
                imag_trap_cnt = imag_trap_cnt + 1
            else:
                hex_dict[(row,col)]=list(tmp)
                
    return hex_dict, hex_pos_grid, trap_dict

"""
creates complementary data set, i.e. a dataset for each
--> year, calender week, hex_cell and species
"""
def complementary_frame(hex_dict, hex_pos_grid):
    
    col_species = pd.Series(list(unique_species)*len(hex_dict)*len(cws)*len(years))
    col_hex_cell = []
    col_x_coor = []
    col_y_coor = []
    for hc in hex_dict:
        col_hex_cell.extend(repeat(hc,len(unique_species)))
        col_x_coor.extend(repeat(hex_pos_grid[hc][1],len(unique_species)))
        col_y_coor.extend(repeat(hex_pos_grid[hc][0],len(unique_species)))
    col_hex_cell = pd.Series(col_hex_cell*len(cws)*len(years))
    col_x_coor = pd.Series(col_x_coor*len(cws)*len(years))
    col_y_coor = pd.Series(col_y_coor*len(cws)*len(years))  
    col_cw = []
    for cw in cws:
        col_cw.extend(repeat(cw,len(unique_species)*len(hex_dict)))
    col_cw = pd.Series(col_cw*len(years))
    col_year = []
    for year in years:
        col_year.extend(repeat(year,len(unique_species)*len(hex_dict)*len(cws)))
    col_year = pd.Series(col_year)
    
    date_dict = {}
    for year in years:
        for cw in cws:
            date_dict[year, cw] = ywd_to_date(year, cw, 2)
    col_date = []
    for i in range(0,col_year.size):
        col_date.extend([date_dict[col_year[i], col_cw[i]]])
    col_date = pd.Series(col_date)
    
    
    new_df = pd.DataFrame(dict(Year = col_year,
                               Date = col_date,
                               Calender_Week = col_cw, 
                               HexCell = col_hex_cell, 
                               Species = col_species, 
                               XCoor = col_x_coor,
                               YCoor = col_y_coor))
    
    return new_df

"""
reshapes training data according to
--> date, hex_cell and species
"""
def train_frame(hex_pos_grid, trap_dict):
    
    series_dates = df_train.loc[:, 'Date']
    series_dates = series_dates.map(str_to_date)
    series_year = series_dates.map(date_to_year)
    series_cw = series_dates.map(date_to_calweek)
    
    series_trap = df_train.loc[:, 'Trap']
    series_hex_cell = []
    series_x_coor = []
    series_y_coor = []
    for el in series_trap:
        hc = trap_dict[el]
        series_hex_cell.extend([hc])
        series_x_coor.extend([hex_pos_grid[hc][1]])
        series_y_coor.extend([hex_pos_grid[hc][0]])
    
    series_hex_cell = pd.Series(series_hex_cell)
    series_x_coor = pd.Series(series_x_coor)
    series_y_coor = pd.Series(series_y_coor)
    
    series_species = df_train.loc[:, 'Species']
    series_numm = df_train.loc[:, 'NumMosquitos']
    series_wnv = df_train.loc[:, 'WnvPresent']
    
    new_df = pd.DataFrame(dict(Year = series_year,
                               Date = series_dates,
                               Calender_Week = series_cw, 
                               HexCell = series_hex_cell, 
                               Species = series_species, 
                               XCoor = series_x_coor,
                               YCoor = series_y_coor,
                               NumMosquitos = series_numm,
                               WnvPresent = series_wnv)) 

    return new_df

"""
reshapes test data according to
--> date, hex_cell and species
"""
def test_frame(hex_pos_grid, trap_dict):   
    
    series_dates = df_test.loc[:, 'Date']
    series_dates = series_dates.map(str_to_date)
    series_year = series_dates.map(date_to_year)
    series_cw = series_dates.map(date_to_calweek)
    
    series_trap = df_test.loc[:, 'Trap']
    series_hex_cell = []
    series_x_coor = []
    series_y_coor = []
    for el in series_trap:
        hc = trap_dict[el]
        series_hex_cell.extend([hc])
        series_x_coor.extend([hex_pos_grid[hc][1]])
        series_y_coor.extend([hex_pos_grid[hc][0]])
    
    series_hex_cell = pd.Series(series_hex_cell)
    series_x_coor = pd.Series(series_x_coor)
    series_y_coor = pd.Series(series_y_coor)    
    
    series_species = df_test.loc[:, 'Species']
    series_id = df_test.loc[:, 'Id']
    
    new_df = pd.DataFrame(dict(Year = series_year,
                               Date = series_dates,
                               Calender_Week = series_cw, 
                               HexCell = series_hex_cell, 
                               Species = series_species, 
                               XCoor = series_x_coor,
                               YCoor = series_y_coor,
                               Id = series_id)) 
    
    return new_df 

    
if __name__ == '__main__':

    print("Starting")
    
    print("Creating Hexagonal Grid")
    hex_dict, hex_pos_grid, trap_dict = create_grid()
    
    print("Creating complementary dataframe")
    df_comp = complementary_frame(hex_dict, hex_pos_grid)
    df_comp.to_csv("COMP/comp_hex.csv", index=False)
    print("Creating train dataframe")
    df_train = train_frame(hex_pos_grid, trap_dict)
    df_train.to_csv("TRAIN/train_hex.csv", index=False)
    print("Creating test dataframe")
    df_test = test_frame(hex_pos_grid, trap_dict)
    df_test.to_csv("TEST/test_hex.csv", index=False)

    
    print("Finished")
    