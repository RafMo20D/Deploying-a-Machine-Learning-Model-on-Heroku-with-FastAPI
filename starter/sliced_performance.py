'''
Checks the sliced data performance

Author: Nícolas
Date: August 16, 2021
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import helper_function


def check_sliced_score():
    df = helper_function.import_data('../data/census_cleaned.csv')
    train, test = train_test_split(df, test_size=0.20, random_state=42)

    trained_model = joblib.load('../model/trainedmodel.joblib')
    encoder = joblib.load('../model/encoder.joblib')
    lb = joblib.load('../model/lb.joblib')
    
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
    splited_items_score = []
    for cat in cat_features:
        for col_item in test[cat].unique():
            df_item = test[test[cat] == col_item]

            X_test, y_test, _, _ = helper_function.process_data(
                df_item,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            
            predicted = trained_model.predict(X_test)

            precision, recall, fbeta = helper_function.compute_model_metrics(y_test, predicted)

            metrics_string = 'Category:{} Item:{}\nPrecision:{} Recall:{} FBeta:{}'.format(
                cat, col_item, precision, recall, fbeta)
            splited_items_score.append(metrics_string)
    
    with open('../model/slice_output.txt', 'w') as f:
        for item_score in splited_items_score:
            f.write(item_score + '\n')

if __name__ == '__main__':
    check_sliced_score()