# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
import helper_function
import joblib

# Add code to load in the data.
data = helper_function.import_data('../data/census_cleaned.csv')
train, test = train_test_split(data, test_size=0.20, random_state=50)

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
X_train, y_train, encoder, lb = helper_function.process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Train and save a model.
model = helper_function.train_model(X_train, y_train)
            
# write the trained model to your workspace in a file called trainedmodel.pkl
joblib.dump(model, "../model/trainedmodel.joblib")
joblib.dump(encoder, "../model/encoder.joblib")
joblib.dump(lb, "../model/lb.joblib")