import mlflow
import mlflow.pyfunc

tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)

# Specify the model name and version
model_name = "Lasso_model"
model_version = "2"  # Change this to the desired version

logged_model = f'models:/{model_name}/{model_version}'
print(logged_model)
try:
    # Load model as a PyFuncModel
    loaded_model = mlflow.pyfunc.load_model(logged_model)
    print(loaded_model)
except Exception as e:
    print("Error loading model:", e)
