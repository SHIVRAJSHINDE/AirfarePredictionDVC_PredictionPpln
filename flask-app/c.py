import mlflow
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


# Set the tracking URI for the MLflow server
tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

# Initialize the MLflow Client
client = MlflowClient(tracking_uri=tracking_uri)

# Search for registered models
registered_models = client.search_registered_models()

# Print registered models and their versions
print("Registered Models:")
for model in registered_models:
    print(f"Model Name: {model.name}")
    for version in model.latest_versions:
        print(f" - Version: {version.version} | Stage: {version.current_stage}")

# Now let's load the model version we are interested in:
model_name = 'Lasso_model'
model_version = 2
model_uri = f"models:/{model_name}/{model_version}"

# Try to load the model
try:
    print("----------------------------------------------------------------")
    print(f"Loading model from URI: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
    print("----------------------------------------------------------------")
    print("Model loaded successfully!")
    print(model)
except Exception as e:
    print("Error loading model:", e)
