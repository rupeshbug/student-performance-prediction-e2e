import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.datascience.entity.config_entity import ModelTrainerConfig
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
import joblib
import mlflow
import mlflow.sklearn
from src.datascience import logger

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/chaulagainrupesh1/end-to-end-ml-project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "chaulagainrupesh1"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c86602b44127e4fdb1576928c0aa2ddbf3118d8d"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.params = config.all_params
        
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]
        
        # Define models with params
        models = {
            "ElasticNet": ElasticNet(
                alpha=self.params["ElasticNet"]["alpha"],
                l1_ratio=self.params["ElasticNet"]["l1_ratio"],
                random_state=self.params["ElasticNet"]["random_state"]
            ),
            "RandomForest": RandomForestRegressor(
                n_estimators=self.params["RandomForest"]["n_estimators"],
                max_depth=self.params["RandomForest"]["max_depth"],
                min_samples_split=self.params["RandomForest"]["min_samples_split"],
                random_state=self.params["RandomForest"]["random_state"]
            ),
            "XGBoost": XGBRegressor(
                n_estimators=self.params["XGBoost"]["n_estimators"],
                max_depth=self.params["XGBoost"]["max_depth"],
                learning_rate=self.params["XGBoost"]["learning_rate"],
                subsample=self.params["XGBoost"]["subsample"],
                random_state=self.params["XGBoost"]["random_state"]
            )
        }
        
        best_model = None
        best_rmse = float("inf")
        all_models = {}
        
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        
        for name, model in models.items():
            with mlflow.start_run(run_name = name):
                model.fit(train_x, train_y)
                
                # predictions
                preds = model.predict(test_x)
                
                # metrics
                rmse, mae, r2 = self.eval_metrics(test_y, preds)
                
                # log metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("r2", r2)
                
                # Log params
                mlflow.log_params(self.params[name])
                
                # Log model to MLflow
                mlflow.sklearn.log_model(model, "model", registered_model_name=f"{name}Model")
                
                # Save individual model locally
                model_path = os.path.join(self.config.root_dir, f"{name}_model.joblib")
                joblib.dump(model, model_path)
                
                # Add model to dictionary for evaluation
                all_models[name] = model
                
                # Update best model
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model_info = {
                        "name": name,
                        "model": model,
                        "rmse": rmse,
                        "path": model_path
                    }
                    
        # Save all models dictionary
        all_models_path = os.path.join(self.config.root_dir, "all_models.joblib")
        joblib.dump(all_models, all_models_path)
        logger.info(f"All models saved at: {all_models_path}")
        
        # Save best model separately
        best_model_path = os.path.join(self.config.root_dir, "best_model.joblib")
        joblib.dump(best_model_info["model"], best_model_path)
        logger.info(f"Best model ({best_model_info['name']}) saved at: {best_model_path}")
        logger.info(f"Best model: {best_model_info['name']} with RMSE={best_model_info['rmse']}")

        return best_model_info
        
                