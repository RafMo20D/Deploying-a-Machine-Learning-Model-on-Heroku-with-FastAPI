import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import starter.helper_function

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

if __name__ == '__main__':
    df = starter.helper_function.import_data('data/census_cleaned.csv')
    train, test = train_test_split(df, test_size=0.20, random_state=42)

    trained_model = joblib.load('model/trainedmodel.joblib')
    encoder = joblib.load('model/encoder.joblib')
    lb = joblib.load('model/lb.joblib')

    X_test, y_test, _, _ = starter.helper_function.process_data(
                df,
                categorical_features=cat_features,
                label="salary", encoder=encoder, lb=lb, training=False)
            
    predicted = trained_model.predict(X_test)

    precision, recall, fbeta = starter.helper_function.compute_model_metrics(y_test, predicted) 