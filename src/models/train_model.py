# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
import pandas as pd
import src.config as cfg
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
#from sklearn.tree import DecisionTreeClassifier
#from src.features import features
from sklearn.linear_model import RidgeClassifier
from catboost import Pool, CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sklearn import linear_model


@click.command()
@click.argument('input_train_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path())
@click.argument('output_predictions_filepath', type=click.Path())

def main(input_train_filepath, input_target_filepath, output_predictions_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('train models')

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
    
    
    #load data
    train = pd.read_pickle(input_train_filepath)
    target = pd.read_pickle(input_target_filepath)
        
    train[CAT_COLS_dr] = train[CAT_COLS_dr].astype('object')
    train_x, train_y = train, target
    
    #models
    ridge = linear_model.Ridge(alpha=.5)
    ridge.fit(train_x, train_y)
    #y_pred_rc = ridge.predict(val_x)


    cat = CatBoostRegressor(iterations=2000)
    cat.fit(train_x, train_y)
    #y_pred_cb = cat.predict(val_x)


    #output
    if not os.path.isdir("models"):
        os.makedirs("models")
        with open(".gitkeep", "w") as _:
            pass    

    pickle.dump(cat, open(output_predictions_filepath +'/catboost.pkl', 'wb'))
    pickle.dump(ridge, open(output_predictions_filepath +'/ridge.pkl', 'wb'))
    




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
