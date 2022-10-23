# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
from preprocess import preprocess_data, preprocess_target, extract_target, cast_types, drop_unnecesary
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import src.config as cfg
from sklearn.preprocessing import LabelEncoder

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_data_filepath', type=click.Path())
@click.argument('output_target_filepath', type=click.Path())
@click.argument('output_dataval_filepath', type=click.Path())
@click.argument('output_targetval_filepath', type=click.Path())

def main(input_filepath, 
         output_data_filepath, output_target_filepath=None, 
         output_dataval_filepath=None, output_targetval_filepath=None):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    df = pd.read_csv(input_filepath)
    
    
    # почему питон не хочет читать обновленный файл конфига? я просто оставлю это здесь 
    
    CAT_COLS_dr = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities',
       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',
       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',
       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
       'Functional', 'GarageType', 'GarageFinish', 'GarageQual',
       'GarageCond', 'PavedDrive',
       'SaleType', 'SaleCondition']
    
    
    df = df.set_index('Id')
    df = drop_unnecesary(df)
    df = df.dropna()
    df = preprocess_data(df)
    df = df.fillna(0)
    df[CAT_COLS_dr] = df[CAT_COLS_dr].astype('category')
    if not os.path.isdir("data/interim"):
        os.makedirs("data/interim")
        with open(".gitkeep", "w") as _:
            pass
        
    encod = LabelEncoder()
    for i in cfg.OHE_COLS:
        df[i] = encod.fit_transform(df[i])
    for i in CAT_COLS_dr:
        df[i] = encod.fit_transform(df[i])
    
    
    if output_target_filepath:
        df, target = df.drop(cfg.TARGET_COLS, axis=1), df[cfg.TARGET_COLS]
        target[config.TARGET_COLS] = target[config.TARGET_COLS].astype(np.int64)
        train_x, val_x, train_y, val_y = train_test_split(df, target, test_size=0.2, 
                                                      shuffle=True, random_state=42, 
                                                      )
        
        
        
        df = train_x
        save_as_pickle(train_y, output_target_filepath)
        save_as_pickle(val_x, output_dataval_filepath)
        save_as_pickle(val_y, output_targetval_filepath)
        
        
    save_as_pickle(df, output_data_filepath)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
