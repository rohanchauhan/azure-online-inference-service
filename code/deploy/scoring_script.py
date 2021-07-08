
import json
import joblib
import numpy as np
from azureml.core import Workspace
from azureml.core.model import Model

# Loads the model
def init():
    global model
    model_path = Model.get_model_path('wine_regression_model')
    model = joblib.load(model_path)
    print('Model Loaded')

# Uses the model to predict new data
def run(new_data):
    data = np.array(json.loads(new_data)['data'])
    predictions = model.predict(np.array(data))
    return json.dumps(predictions.tolist())
