import os
import json

import mlflow
import time
import dagshub
from mlflow.tracking import MlflowClient


class ModelManager:
    """Class to manage model saving, loading, and registration with MLflow."""

    def __init__(self, model_name: str, info_path: str):
        self.model_name = model_name
        self.info_path = info_path
        #self.client = mlflow.tracking.MlflowClient()
        self.client = MlflowClient()
        mlflow.set_tracking_uri('https://dagshub.com/SHIVRAJSHINDE/AirfarePredictionDVC_PredictionPpln.mlflow')
        dagshub.init(repo_owner='SHIVRAJSHINDE',repo_name='AirfarePredictionDVC_PredictionPpln',mlflow=True)
        # self.tracking_uri = "http://localhost:5000"
        # mlflow.set_tracking_uri(self.tracking_uri)

    def save_model_info(self, run_id: str, model_path: str) -> None:
        """Save the model run ID and path to a JSON file."""
        os.makedirs(os.path.dirname(self.info_path), exist_ok=True)  # Ensure the directory exists
        model_info = {'run_id': run_id, 'model_path': model_path}
        
        with open(self.info_path, 'w') as file:
            json.dump(model_info, file, indent=4)

    def load_model_info(self) -> dict:
        """Load the model info from a JSON file."""
        if not os.path.exists(self.info_path):
            raise FileNotFoundError(f"Model info file not found: {self.info_path}")

        with open(self.info_path, 'r') as file:
            return json.load(file)

    def register_model(self):
        """Register the model with the MLflow Model Registry."""
        model_info = self.load_model_info()
        print("----------------------------------------------------------------")
        print(model_info)
        print("----------------------------------------------------------------")
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        print("----------------------------------------------------------------")
        print(model_uri)
        print("----------------------------------------------------------------")
        # Register the model
        model_version = mlflow.register_model(model_uri, self.model_name)
        print("----------------------------------------------------------------")
        print(model_version)
        print("----------------------------------------------------------------")
        print("model_name,model_version.version")
        print(self.model_name,model_version.version)
        # Transition the model to "Staging" stage
        

if __name__ == '__main__':
    model_name="Lasso_model"
    info_path='reports/experiment_info.json'

    model_manager = ModelManager(model_name, info_path)
    
    # Load model info and register
    model_manager.register_model()












