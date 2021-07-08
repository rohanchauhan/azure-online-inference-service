# Azure Realtime Inference Service
In this project, we create a web service which can be consumed using the REST endpoint and authentication key. In this project, we used the wine quality dataset and used a simple linear regression model for demonstration.

## Code Structure
| **File**                                | **Description**                                |
|-----------------------------------------|------------------------------------------------|
| `data/winequality-white.csv`            | Wine quality dataset from UCI                  |
| `code/train/training_environment.yml`   | Conda file for training environment            |
| `code/train/train.py`                   | Trains model and saves the model and test data |
| `code/train/script_run_config.py`       | Runs training script and registers the model   |
| `code/deploy/inference_environment.yml` | Conda file for inference environment           |
| `code/deploy/scoring_script.py`         | Uses register model to predict test data       |
| `code/deploy/deployment_script.py`      | Deploys web service on Docker (ACI)            |
| `code/test/test_using_sdk.py`           | Tests web service using only Azure SDK         |
| `code/test/test_using_http.py`          | Tests web service using HTTP request           |

## How to Run
1. Clone this repository in Azure ML workspace using terminal. Use https, if you haven't added SSH keys to Github.
2. Run `script_run_config.py` -> Used to train the model and register it
3. Run `deployment_script.py` -> Register model is deployed as endpoint 
4. Run `test_using_sdk.py` -> Consumes service using SDK
5. Run `test_using_http.py` -> Consumes service using HTTP request

## Changes
1. The web service is deployed on Azure Container. For production, we can deploy it on Azure Kubernetes Service by making the following changes in `code/deploy/deployment_script.py`:
```{python}
compute_name="aks-cluster"
compute_config=AksCompute.provisioning_configuration(location=""central-india"")
production_cluster = ComputeTarget.create(ws, compute_name, compute_config)
production_cluster.wait_for_completion(show_output=True)
from azureml.core.webservice import AksWebservice

production_config = AksWebservice.deploy_configuration(autoscale_enabled=False, num_replicas=3, cpu_cores=2, memory_gb=4)

myenv = Environment.from_conda_specification(name="myenv", file_path="myenv.yml")
inference_config = InferenceConfig(entry_script="score.py", environment=myenv)

service_name = "production-model"
model = Model(ws, "production-model")
aks_service = Model.deploy(ws,
                           models=[model],
                           inference_config=inference_config,
                           deployment_config=production_config,
                           deployment_target=production_cluster,
                           name=service_name)

aks_service.wait_for_deployment(show_output=True)
print(aks_service.state)
```

2. For training the model, we used the development cluster itself. A separate training cluster should be created for training model with larger datasets by making the following changes in `code/train/script_run_config.py`
```{python}
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Compute params
compute_name = 'rohan-vm-cluster'
training_cluster = None

if compute_name in ws.compute_targets:
    training_cluster = ComputeTarget(ws, compute_name)
    print("Using existing cluster.")
else:
    try:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size ='STANDARD_DS11_V2', 
            max_nodes=2 )
        training_cluster = ComputeTarget.create(ws, compute_name, compute_config)
        training_cluster.wait_for_completion(show_output=True)
    except Exception as ex:
        print(ex)
    print("Cluster created.")
```
