import pandas as pd

# Load train and test set
df_train = pd.read_csv('TRAIN/train_unified_traps.csv')

train = []

for sub in df_train.groupby(['Date', 'Trap', 'Species']):
    entry = sub[1].iloc[0]
    entry.NumMosquitos = sub[1].NumMosquitos.sum()
    entry.WnvPresent = sub[1].WnvPresent.max()
    train.append(entry)
    
df_train_new = pd.DataFrame(train)

# Save corrected train and test sets
df_train_new.to_csv("TRAIN/train_dropped.csv", index=False)