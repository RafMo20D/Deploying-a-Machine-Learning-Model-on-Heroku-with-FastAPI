# Model Card

The predictive model is used to predict salaries using a classification model trained with logistic regression on publicly available Census Bureau data.

## Model Details
Nícolas Pauli created the model. It is logistic regression using the default hyperparameters in scikit-learn 0.24.2.
## Intended Use
This model should be used to predict salaries using a classification model on publicly available Census Bureau data.
## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/census+income). 

The original data set has 32561 rows, and a 80-20 split was used to break this into a train and test set. It was droped the rows with '?' and the columns "fnlgt", "capital-gain" and "capital-loss" that did not add much to the predictability. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.
## Evaluation Data
The evaluation data set followed the same colums from the training data and was a 80-20 split was used to break this into a train and test set.
## Metrics
The model was evaluated using precision, recall and F-Beta score. 

## Ethical Considerations
All the rights from the dataset is reserved to UCI and should be refenced.
## Caveats and Recommendations
The original dataset has whitespaces that can cause misleaded predictions considering some items unique during the One Hot Enconding process so it is import to clean the dataset.