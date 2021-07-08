from azureml.core import Workspace, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

# Get workspace
ws = Workspace.from_config()

# Create Inference config
inference_config = InferenceConfig(
    runtime='python',
    entry_script='scoring_script.py',
    conda_file='inference_environment.yml')

# Create deployment target
deployment_config = AciWebservice.deploy_configuration(cpu_cores = 1,
    memory_gb = 1)

# Create Service
service_name='realtime-wine-inference-service'
regression_model = Model(ws,'wine_regression_model')

service = Model.deploy(workspace=ws, 
    name=service_name,
    models=[regression_model],
    inference_config=inference_config,
    deployment_config=deployment_config)

service.wait_for_deployment(True)
print(service.state)

# Write the endpoint for use in testing
endpoint = service.scoring_uri
with open('./../../data/endpoint.txt', 'w') as f:
    f.write(endpoint)
