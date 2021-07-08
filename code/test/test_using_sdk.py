import pandas as pd 
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

# Get service
service_name = 'realtime-wine-inference-service'
service = Webservice(workspace=ws,name=service_name)

# Convert data to json
json_data = json.dumps({"data": X_test})

# Call the service
predictions = service.run(input_data = json_data)
y_pred = json.loads(predictions)
for x in range(len(y_pred)):
    print('Wine {} : | Predicted Quality: {} | True Quality: {}'.format(x,y_pred[x],y_test[x]))
