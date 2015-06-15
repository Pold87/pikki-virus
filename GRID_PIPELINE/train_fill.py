import pandas as pd
import numpy as np

# Load reshaped training dataset
df_train = pd.read_csv("TRAIN/train_hex.csv")
# Find unique species
all_species = set(df_train.Species.unique())

# For each date and hex_cell create additional rows for all species
# not accounted for in the training dataset with 
# NumMosquitos=0 and WnvPresent=0
all_series = []
for sub in df_train.groupby(['Date', 'HexCell']):
    species_in = set(sub[1].Species.unique())
    species_miss = list(all_species - species_in)
    for miss in species_miss:
        tmp = sub[1].iloc[0].copy()
        tmp.Species = miss
        tmp.NumMosquitos = 0
        tmp.WnvPresent = 0
        all_series.append(tmp)
  
# Combine original dataframe with additional rows      
new_frame = pd.DataFrame(all_series)
df_new_train = df_train.append(new_frame).sort().reset_index()
df_new_train = df_new_train.drop(['index'],axis=1)

# Save filled training dataset
df_new_train.to_csv("TRAIN/train_hex_filled.csv", index=False)