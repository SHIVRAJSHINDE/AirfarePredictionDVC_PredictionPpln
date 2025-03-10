import mlflow
import dagshub
import mlflow.pyfunc


mlflow.set_tracking_uri('https://dagshub.com/SHIVRAJSHINDE/AirfarePredictionDVC_PredictionPpln.mlflow')
dagshub.init(repo_owner='SHIVRAJSHINDE',repo_name='AirfarePredictionDVC_PredictionPpln',mlflow=True)


model_uri = "runs:/a1b2fb117c4c4c909bf9f79822b70836/model"
model = mlflow.pyfunc.load_model(model_uri)