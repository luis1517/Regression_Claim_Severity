# Insurance Claim Severity Regression Model
This repository contains a regression model to predict the severity of insurance claims. The model is based on historical data from an insurance company and uses a variety of features related to the claim, such as the type of claim, the policyholder's age, and the location of the incident.

## Getting Started
To get started with this model, you'll need to have the following software installed on your machine:

- Python 3.6 or higher
- Scikit-learn library
- Xgboost library
- CatBoost libary
- Pandas library
- Numpy library
- Seaborn library

You can install these libraries using pip:

```bash
  pip install scikit-learn pandas numpy Xgboost CatBoost seaborn
```
## Data
The data used to train and test the model is contained in the data folder. The train.csv file contains the training data, while the test.csv file contains the testing data. Both files have the same format, with each row representing a single claim and each column representing a different feature. The target variable, severity, is included in the training data but not in the testing data.

## Model
The regression model is built using Python 3 and the scikit-learn machine learning library. The model uses a combination of linear and non-linear regression techniques to predict the severity of insurance claims. The model is trained on the training data and evaluated on the testing data using mean squared error (MSE) as the performance metric.

## Files
data/train.csv: The training data used to train the model.
data/test.csv: The testing data used to evaluate the model.
model/train_model.py: The Python script used to train the regression model.
model/predict.py: The Python script used to predict claim severity on new data.
Usage
To train the model, run the train_model.py script in the model directory. This will generate a trained model file named severity_model.pkl. To predict claim severity on new data, run the predict.py script in the model directory and pass in the path to the new data file as a command line argument. The script will load the trained model from severity_model.pkl and generate predictions for each row in the new data.

## Conclusion
This repository contains a regression model to predict the severity of insurance claims based on historical data. The model is built using Python and scikit-learn and can be trained on new data using the provided train_model.py script. Once trained, the model can be used to generate predictions on new data using the predict.py script.  
