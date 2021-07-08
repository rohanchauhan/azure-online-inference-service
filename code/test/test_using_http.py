import pandas as pd 
import requests
import json
from azureml.core import Workspace, Webservice


# Download data from datastore
ws = Workspace.from_config()
default_ds = ws.get_default_datastore()
default_ds.download(
    target_path='.',
    prefix='wine-quality/test-data/'
)

# Read test data
X_test = pd.read_csv('wine-quality/test-data/X_test.csv',index_col=False).to_numpy().tolist()
y_test = pd.read_csv('wine-quality/test-data/y_test.csv',index_col=False).to_numpy().tolist()

# Read endpoint
with open('./../../data/endpoint.txt', 'r') as f:
    endpoint = f.read()

# Convert data to json
json_data = json.dumps({"data": X_test})

# Set the content type
headers = { 'Content-Type':'application/json' }

predictions = requests.post(endpoint, json_data, headers = headers)
print(predictions)
y_pred = json.loads(predictions.json())

for x in range(len(y_pred)):
    print('Wine {} : | Predicted Quality: {} | True Quality: {}'.format(x,y_pred[x],y_test[x]))
