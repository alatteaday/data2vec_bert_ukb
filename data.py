import pandas as pd
from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/Demo_Ins0_final.csv', low_memory=False)

pre_train, pre_val = train_test_split(df, test_size=0.1, shuffle=True)

pre_train.to_csv('./data/pre_train.csv', index=False)
pre_val.to_csv('./data/pre_val.csv', index=False)

