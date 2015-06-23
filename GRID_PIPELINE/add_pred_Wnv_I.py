import pandas as pd
import numpy as np

import multiprocessing as mp
import pandas.util.testing as pdt

def mean_num(chunk):
    
    indices_0=np.array([[0,0]])
    indices_1=np.array([[-1,-1],[-1,0],[0,-1],[0,1],[1,-1],[1,0]])    
    indices_2=np.array([[-2,-1],[-2,0],[-2,1],[-1,-2],[-1,1],[0,-2],[0,2],[1,-2],[1,1],[2,-1],[2,0],[2,1]])
             
    hex_c = np.array(eval(chunk.iloc[0].HexCell))

    hex_list_0 = indices_0 + hex_c
    hex_list_1 = indices_1 + hex_c
    hex_list_2 = indices_2 + hex_c
    
    new_array_0 = [tuple(row) for row in hex_list_0]
    new_array_1 = [tuple(row) for row in hex_list_1]
    new_array_2 = [tuple(row) for row in hex_list_2]
    
    mask = mask_dict[chunk.iloc[0].Year_x, chunk.iloc[0].Species, chunk.iloc[0].Calender_Week]
    
    tmp_1 = new_df[mask]
    sumit_0 = []
    sumit_1 = []
    sumit_2 = []
    
    for it in new_array_0:
        tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
        sumit_0.append(tmp_2.iloc[0].WnvPresent)
    for it in new_array_1:
        tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
        try:
            sumit_1.append(tmp_2.iloc[0].WnvPresent)
        except:
            pass
    for it in new_array_2:
        tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
        try:
            sumit_2.append(tmp_2.iloc[0].WnvPresent)
        except:
            pass
        
    chunk['Mean_Wnv_Lvl_0'] = np.mean(sumit_0)
    chunk['Min_Wnv_Lvl_0'] = np.min(sumit_0)
    chunk['Max_Wnv_Lvl_0'] = np.max(sumit_0)
    chunk['Min_Wnv_Lvl_0'] = np.min(sumit_0)
    chunk['Max_Wnv_Lvl_0'] = np.max(sumit_0)
    chunk['Mean_Wnv_Lvl_1'] = np.mean(sumit_1)
    chunk['Min_Wnv_Lvl_1'] = np.min(sumit_1)
    chunk['Max_Wnv_Lvl_1'] = np.max(sumit_1)
    chunk['Mean_Wnv_Lvl_2'] = np.mean(sumit_2)
    chunk['Min_Wnv_Lvl_2'] = np.min(sumit_2)
    chunk['Max_Wnv_Lvl_2'] = np.max(sumit_2)
    
    if chunk.iloc[0].Calender_Week > 19:
        mask = mask_dict[chunk.iloc[0].Year_x, chunk.iloc[0].Species, (chunk.iloc[0].Calender_Week-1)]
        tmp_1 = new_df[mask]
        
        sumit_0 = []
        sumit_1 = []
        sumit_2 = []
        
        for it in new_array_0:
            tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
            sumit_0.append(tmp_2.iloc[0].WnvPresent)
        for it in new_array_1:
            tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
            try:
                sumit_1.append(tmp_2.iloc[0].WnvPresent)
            except:
                pass
        for it in new_array_2:
            tmp_2 = tmp_1[tmp_1['HexCell']==str(it)]
            try:
                sumit_2.append(tmp_2.iloc[0].WnvPresent)
            except:
                pass
            
        chunk['Mean_Wnv_Lvl_0_cwbef'] = np.mean(sumit_0)
        chunk['Min_Wnv_Lvl_0_cwbef'] = np.min(sumit_0)
        chunk['Max_Wnv_Lvl_0_cwbef'] = np.max(sumit_0)
        chunk['Mean_Wnv_Lvl_1_cwbef'] = np.mean(sumit_1)
        chunk['Min_Wnv_Lvl_1_cwbef'] = np.min(sumit_1)
        chunk['Max_Wnv_Lvl_1_cwbef'] = np.max(sumit_1)
        chunk['Mean_Wnv_Lvl_2_cwbef'] = np.mean(sumit_2)
        chunk['Min_Wnv_Lvl_2_cwbef'] = np.min(sumit_2)
        chunk['Max_Wnv_Lvl_2_cwbef'] = np.max(sumit_2)
        
    else:
        chunk['Mean_Wnv_Lvl_0_cwbef'] = 0
        chunk['Min_Wnv_Lvl_0_cwbef'] = 0
        chunk['Max_Wnv_Lvl_0_cwbef'] = 0
        chunk['Mean_Wnv_Lvl_1_cwbef'] = 0
        chunk['Min_Wnv_Lvl_1_cwbef'] = 0
        chunk['Max_Wnv_Lvl_1_cwbef'] = 0
        chunk['Mean_Wnv_Lvl_2_cwbef'] = 0
        chunk['Min_Wnv_Lvl_2_cwbef'] = 0
        chunk['Max_Wnv_Lvl_2_cwbef'] = 0
           
    return chunk

def load_apply(ndf):
    
    res = ndf.groupby(['Year_x', 'Calender_Week', 'HexCell', 'Species']).apply(mean_num)
    
    return res

def load_numbers():

    newp_df = new_df.copy()
    #newp_df = new_df.iloc[0:300].copy()
    newp_df = newp_df.reset_index()    
    
    cpus = 50
    
    p = mp.Pool(processes=cpus)
    split_dfs = np.array_split(newp_df,cpus)
    pool_results = p.map(load_apply, split_dfs)
    p.close()
    p.join()  
    
    parts = pd.concat(pool_results, axis=0)
    
    pdt.assert_series_equal(parts['index'], newp_df['index'])

    parts = parts.drop(['WnvPresent', 'index'], axis=1)

    return parts

def load_data(path_train='train.csv', path_comp='comp.csv', path_test='test.csv'):

    """
    Read in train.csv or test.csv and preprocess the data frame
    """

    # read in weather data
    df_numbers = load_numbers()

    df_train = pd.read_csv(path_train) 
    merged_df_train = pd.merge(df_train, df_numbers, on=['Year_x', 'Calender_Week', 'HexCell', 'Species'])
    df_train = None
    df_comp = pd.read_csv(path_comp)   
    merged_df_comp = pd.merge(df_comp, df_numbers, on=['Year_x', 'Calender_Week', 'HexCell', 'Species'])
    df_comp = None
    df_test = pd.read_csv(path_test)   
    merged_df_test = pd.merge(df_test, df_numbers, on=['Year_x', 'Calender_Week', 'HexCell', 'Species'])
    df_test = None


    return merged_df_train, merged_df_comp, merged_df_test

"""

"""
df_num = pd.read_csv('COMP/comp_hex_wnv_I.csv')  
series_year = df_num['Year_x']
series_species = df_num['Species']
series_hex_cell = df_num['HexCell']
series_cw = df_num['Calender_Week']
series_wnv = df_num['WnvPresent']

new_df = pd.DataFrame(dict(Year_x = series_year,
                           Calender_Week = series_cw, 
                           HexCell = series_hex_cell, 
                           Species = series_species, 
                           WnvPresent = series_wnv)) 
    
mask_dict = {}                      
for y in series_year.unique():
    for s in series_species.unique():
        for c in series_cw.unique():
            mask_dict[y,s,c]= (new_df['Year_x']==y) & (new_df['Species']==s) & (new_df['Calender_Week']==c)

df_num = None
series_year = None
series_species = None
series_hex_cell = None
series_cw = None
series_numm = None


if __name__ == '__main__':

    print("Starting")
    print("Processing Training Data")
    # read in training data
    df_train, df_comp, df_test = load_data("TRAIN/train_hex_pred_II.csv", "COMP/comp_hex_pred_II.csv", "TEST/test_hex_pred_II.csv")
    df_train.to_csv("TRAIN/train_hex_wnv_pred_I.csv", index=False)
    df_comp.to_csv("COMP/comp_hex_wnv_pred_I.csv", index=False)
    df_test.to_csv("TEST/test_hex_wnv_pred_I.csv", index=False)
    
    print("Finished")
