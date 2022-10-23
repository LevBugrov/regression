# -*- coding: utf-8 -*-
from cgi import test
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import pickle
from src.data.preprocess import *
import src.config as cfg
from sklearn.preprocessing import LabelEncoder
import os
import features as fe


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_catboost_model', type=click.Path())
@click.argument('input_sklearn_model', type=click.Path())
@click.argument('output_predictions_filepath', type=click.Path())

def main(input_data_filepath, input_catboost_model, input_sklearn_model, output_predictions_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # load data
    test_data = pd.read_csv(input_data_filepath)
    test_data = preprocess_data(test_data)
    
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
    
    encod = LabelEncoder()
    for i in cfg.OHE_COLS:
        test_data[i] = encod.fit_transform(test_data[i])
    for i in CAT_COLS_dr:
        test_data[i] = encod.fit_transform(test_data[i])

    # feature engineering
    test_data = fe.add_is_porch(test_data)
    test_data = fe.add_num_of_bath(test_data)

    sklearn_model = pickle.load(open(input_sklearn_model, 'rb'))
    catboost_model = pickle.load(open(input_catboost_model, 'rb'))

    catboost_prediction = catboost_model.predict(test_data)
    sklearn_prediction = sklearn_model.predict(test_data)

    df_pred_sc = pd.DataFrame(sklearn_prediction, columns = cfg.TARGET_COLS, index=test_data.index)
    df_pred_cb = pd.DataFrame(catboost_prediction, columns = cfg.TARGET_COLS, index=test_data.index)
    
    
    if not os.path.isdir("reports/inference"):
        os.makedirs("reports/inference")
        
    df_pred_sc.to_csv(output_predictions_filepath + 'sklearn_pred.csv')
    df_pred_cb.to_csv(output_predictions_filepath + 'catboost_pred.csv')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()