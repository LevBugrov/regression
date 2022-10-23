import pandas as pd
import numpy as np
import src.config as cfg

def add_is_porch(df: pd.DataFrame) -> pd.DataFrame:
    for a,b,c,d in zip(df['OpenPorchSF'].values, df['EnclosedPorch'].values, df['3SsnPorch'].values, df['ScreenPorch'].values):
        df["is_porch"] = any([a,b,c,d])
    
    cfg.OHE_COLS.append("is_porch")
    return df 

def add_num_of_bath(df: pd.DataFrame) -> pd.DataFrame:
    for a,b,c,d in zip(df['BsmtFullBath'].values, df['BsmtHalfBath'].values, df['FullBath'].values, df['HalfBath'].values):
        df["num_of_bath"] = a+b+c+d

    
    cfg.OHE_COLS.append("num_of_bath")
    return df
