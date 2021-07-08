
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies

# Get workspace
ws = Workspace.from_config()

# Create environment
env = Environment.from_conda_specification("wine_train_env", 'training_environment.yml')

# Get the default datastore
default_ds = ws.get_default_datastore()

dataset_name = 'wine quality dataset'
if dataset_name not in ws.datasets:
    default_ds.upload_files(
        files=['../../data/winequality-white.csv'],
        target_path='wine-quality/data/',
        overwrite=True,
        show_progress=True)
    
    tab_dataset = Dataset.Tabular.from_delimited_files(
        path=(default_ds,'wine-quality/data/*.csv'),
        separator=';')
        
    try:
        tab_dataset = tab_dataset.register(workspace=ws, 
                                name=dataset_name,
                                description='wine quality data from UCI',
                                tags = {'format':'CSV'},
                                create_new_version=True)
        print('Dataset registered.')
    except Exception as ex:
        print(ex)
else:
    tab_dataset = Dataset.get_by_name(ws, dataset_name)
    print('Dataset already registered.')

# Create script config
script_config = ScriptRunConfig(
    source_directory='.',
    script='train.py',
    arguments=['--input-data',tab_dataset.as_named_input('training_data')],
    environment=env)

print('Starting Experiment')
# Run experiment
experiment_name='wine-regression-experiment'
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

# Wait for experiment to complete
run.wait_for_completion()
print('Experiment ended')


# Register the model
run.register_model(
    model_path='outputs/regression_model.pkl',
    model_name='wine_regression_model',
    tags={'Training context':'Realtime Inference Pipeline'})
print('Model Registered')
