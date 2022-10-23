import pandas as pd
import numpy as np
import src.config as config


def drop_unnecesary(df: pd.DataFrame) -> pd.DataFrame:
    DROP_COLS = ['Alley', 'BsmtFinType2', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature']
    for cols in DROP_COLS:
        if cols in df.columns:
            df = df.drop(cols, axis=1)
            
        
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    #df = df.set_index('Id')
    return df

def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[config.TARGET_COLS] = df[config.TARGET_COLS].astype(np.int8)
    return df

def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(config.TARGET_COLS, axis=1), df[config.TARGET_COLS]
    return df, target