import mlflow
import mlflow.pyfunc
tracking_uri = "http://localhost:5000"
mlflow.set_tracking_uri(tracking_uri)


logged_model = 'runs:/ab8dfef082cb48aa915ceeedec273f9e/Lasso_model'

try:
    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)



    print(loaded_model)
except Exception as e:
    print("Error loading model:", e)
