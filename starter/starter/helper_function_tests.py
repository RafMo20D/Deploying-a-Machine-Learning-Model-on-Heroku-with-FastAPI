'''
Testing & Logging for churn_library.py script

Author: Nícolas
Date: August 16, 2021
'''
import os
import logging
import pytest
import joblib
import starter.helper_function


@pytest.fixture
def data():
    """
    Get the cleaned dataset
    """
    df = starter.helper_function.import_data("../data/census_cleaned.csv")
    return df

def test_import_data(data):
    '''
    test import_data 
    '''
    assert data.shape[0] > 0
    assert data.shape[1] > 0
    

def test_process_data(data):
    """
    test process_data function
    """
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    X_test, y_test, _, _ = starter.helper_function.process_data(
        data, categorical_features=cat_features, label="salary", training=True)
       
    assert len(X_test) == len(y_test)

def test_columns_name(data):
    """
    test the columns dataset name
    """
    right_columns = [
        "age",
        "workclass",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
        "salary"
    ]

    obtained_columns = data.columns.values
    assert list(right_columns) == list(obtained_columns)
    
