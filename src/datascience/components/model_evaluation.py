import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import numpy as np
import joblib

from src.datascience.utils.common import save_json
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.constants import *
from src.datascience.utils.common import read_yaml, create_directories, save_json

import os

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/chaulagainrupesh1/end-to-end-ml-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "chaulagainrupesh1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c86602b44127e4fdb1576928c0aa2ddbf3118d8d"

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2
    
    def log_into_mlflow(self):

        test_data = pd.read_csv(self.config.test_data_path)
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        # Load all trained models
        model_dict = joblib.load(self.config.model_path)

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Evaluate each model
        for model_name, model in model_dict.items():
            print(f"\n🔍 Evaluating model: {model_name}")

            params = self.config.all_params.get(model_name, {})

            with mlflow.start_run(run_name=f"{model_name}_evaluation"):
                preds = model.predict(test_x)
                rmse, mae, r2 = self.eval_metrics(test_y, preds)

                # Save metrics locally
                metrics = {"rmse": rmse, "mae": mae, "r2": r2}
                local_metric_file = Path(str(self.config.metric_file_name).replace(".json", f"_{model_name}.json"))
                save_json(path=local_metric_file, data=metrics)

                # Log parameters and metrics to MLflow
                mlflow.log_params(params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)

                # Log model to MLflow
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=model_name
                    )
                else:
                    mlflow.sklearn.log_model(model, "model")

            print(f"✅ Logged {model_name} to MLflow successfully.")
    


