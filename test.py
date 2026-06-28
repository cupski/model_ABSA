import pandas as pd
from preprocessing_functions import compare_distribution
    
ASPECT_DOMAIN = ['fuel', 'machine', 'others', 'part', 'price', 'service']
df_train = pd.read_csv("train_preprocess.csv")
df_val = pd.read_csv("valid_preprocess.csv")
df_test = pd.read_csv("test_preprocess.csv")
aspect_cols = [f'{asp}' for asp in ASPECT_DOMAIN]
for col in aspect_cols:
    compare_distribution(df_train, df_val, df_test, col)