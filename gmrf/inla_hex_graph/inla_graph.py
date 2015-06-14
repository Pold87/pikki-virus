#from pandas import io
import pandas as pd
import numpy as np
#import re
#from scipy import stats
from math import sin, cos, sqrt, atan2, radians
import datetime as dt
from scipy.spatial import distance
import matplotlib as plt

LATI_ZERO = 41.5
LONG_ZERO = -88.0
LATI_END  = 42.1
LONG_END  = -87.4
HEX_CELL_SIZE  = 1.0
HEX_CELL_HEIGHT = HEX_CELL_SIZE * 2.0
HEX_CELL_WIDTH = sqrt(3.0) / 2.0 * HEX_CELL_HEIGHT
HEX_CELL_VERT = HEX_CELL_HEIGHT * 3.0 / 4.0
HEX_CELL_HORI = HEX_CELL_WIDTH

df_train = pd.read_csv('train_trap_correct.csv')
df_test = pd.read_csv('test_trap_correct.csv')

trap_series = df_train.loc[:, 'Trap'].append(df_test.loc[:, 'Trap'])
unique_traps = trap_series.unique()

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

def neighbour_cell(hex_grid, level):
    indices=[[0,0]]
    if level>=1:
        indices=indices + [[-1,-1],[-1,0],[0,-1],[0,1],[1,-1],[1,0]]
    if level>=2:
        indices=indices + [[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,1],[0,-2],[0,2],[1,-2],[1,1],[2,-1],[2,0],[2,1]]
    if level>=3:
        indices=indices + [[-3,-1],[-3,0],[-3,1],[-2,-2],[-2,2],[-1,-3],[-1,2],[0,-3],[0,3],[1,-3],[1,2],[2,-2],[2,2],[3,-1],[3,0],[3,1]]
    indices = np.array(indices)
    cell_neigh={}
    for index, el in hex_grid.iterrows():
        tmp_neigh = []
        for ind in indices:
            row = el[0]+ind[0]
            col = el[1]+ind[1]
            tmp=list(hex_grid[hex_grid[0]==row][hex_grid[1]==col].index.values)
            if index in tmp:
                tmp.remove(index)
            tmp_neigh = tmp_neigh+tmp

        cell_neigh[index]=list(tmp_neigh)
    return cell_neigh# list of neighbouring cells

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

    for i, el in enumerate(unique_traps):
        ind = df_test[df_test['Trap'] == el].index.tolist()[0]
        lat1 = df_test['Latitude'].loc[ind]
        long1 = df_test['Longitude'].loc[ind]
        hex_c = hex_cell(hex_pos_grid, lat1,long1)
        hex_grid[i]=hex_c

    hex_dict={}

    row_min = (np.min(hex_grid[:,0])-2).astype(int)
    row_max = (np.max(hex_grid[:,0])+2).astype(int)
    col_min = (np.min(hex_grid[:,1])-2).astype(int)
    col_max = (np.max(hex_grid[:,1])+2).astype(int)

    hex_grid_2 = pd.DataFrame(data=hex_grid, index=unique_traps)


    for row in range(row_min, row_max+1):
        for col in range(col_min, col_max+1):
            tmp=list(hex_grid_2[hex_grid_2[0]==row][hex_grid_2[1]==col].index.values)
            hex_dict[(row,col)]=list(tmp)

    return hex_grid, hex_dict, hex_pos_grid

hex_grid, hex_dict, hex_pos_grid = create_grid()

hex_df = pd.DataFrame(data=hex_grid, index=unique_traps)
# neigh_0 = neighbour_cell(hex_df,0)
# neigh_1 = neighbour_cell(hex_df,1)
# neigh_2 = neighbour_cell(hex_df,2)
neigh_3 = neighbour_cell(hex_df,3)

df_train = df_train[1:100]
df_test = df_test[1:100]

trap_series = df_train.loc[:, 'Trap'].append(df_test.loc[:, 'Trap'])
unique_traps = trap_series.unique()

list_traps = list(unique_traps)

df_train['TrapID'] = -1
for idx, row in df_train.iterrows():
    df_train['TrapID'][idx] = list_traps.index(df_train['Trap'][idx])

df_test['TrapID'] = -1
for idx, row in df_test.iterrows():
    df_test['TrapID'][idx] = list_traps.index(df_test['Trap'][idx])

mapping = pd.DataFrame({ 'TrapID' : df_train.loc[:, 'TrapID'].append(df_test.loc[:, 'TrapID']),
    'Latitude' : df_train.loc[:, 'Latitude'].append(df_test.loc[:, 'Latitude']),
    'Longitude' : df_train.loc[:, 'Longitude'].append(df_test.loc[:, 'Longitude'])})
mapping = mapping.drop_duplicates('TrapID')

mappingfile = open('./hex_mapping.csv', 'w+')
mappingfile.write("Latitude, Longitude, TrapID")
for idx, row in mapping.iterrows():
    mappingfile.write("%f, %f, %d\n" % (mapping['Latitude'][idx], mapping['Longitude'][idx], mapping['TrapID'][idx]))


graphfile = open('./hex.graph', 'w+')
for row in neigh_3:
    graphfile.write("%d %d %s\n" % (list_traps.index(row), len(neigh_3[row]), ' '.join([str(neighbour) for neighbour in neigh_3[row]])))
