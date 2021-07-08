
from azureml.core import Run, Experiment, Model, Datastore, Workspace
import pandas as pd 
import numpy as np
import joblib
from sklearn.model_selection import train_test_split 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import argparse
import os

# Get dataset as argument
parser = argparse.ArgumentParser()
parser.add_argument("--input-data", type=str, dest='dataset_id', help='training dataset')
args=parser.parse_args()

# Get the experiment run context
run = Run.get_context()

print('Loading Dataset')
# Load Dataset
wine_quality = run.input_datasets['training_data'].to_pandas_dataframe()
X, y = wine_quality.drop('quality', axis=1), wine_quality.quality

# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

# Write X_test, y_test for later on testing service
os.makedirs('outputs', exist_ok=True)
X_test.to_csv('outputs/X_test.csv',index=False, index_label=False)
y_test.to_csv('outputs/y_test.csv',index=False, index_label=False)

# Upload test data to datastore
ws = Workspace.from_config()
default_ds = ws.get_default_datastore()
default_ds.upload_files(
    files=['outputs/X_test.csv','outputs/y_test.csv'],
    target_path='wine-quality/test-data/',
    overwrite=True,
    show_progress=True)

print('Training Model')
# Train Model
regression_model = linear_model.LinearRegression().fit(X_train,y_train)

# Mean Squared Error and R Squared Score
y_pred = regression_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print & Log metrics
run.log('MSE', mse)
print("Mean Squared Error: {:.2f}".format(mse))

run.log('R2', r2)
print("R Squared : {:.2f}".format(r2))

print('Storing Model')
# Store the model
joblib.dump(value=regression_model, filename='outputs/regression_model.pkl')

# Complete the run
run.complete()
