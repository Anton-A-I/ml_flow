import mlflow
import os
import pandas as pd

from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

os.getenv("MLFLOW_TRACKING_URI", "No env")
mlflow.get_registry_uri()
exp_id = mlflow.create_experiment(name="anton_ivanov")
mlflow.set_experiment(experiment_name="anton_ivanov")
mlflow.end_run() 
#with mlflow.start_run():
    # Обучим модель.
    
housing = fetch_california_housing(as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(housing['data'], housing['target'])
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5)
#model = LinearRegression()
models = dict(zip(["RandomForest", "GradientBoosting", "LinearRegression"], 
              [RandomForestRegressor(), GradientBoostingRegressor(), LinearRegression()]))
with mlflow.start_run(run_name="anton_a_i", experiment_id=exp_id, description = "parent") as parent_run:
    for model_name in models.keys():
    # Запустим child run на каждую модель.
        with mlflow.start_run(run_name=model_name, experiment_id=exp_id, nested=True) as child_run:
            model = models[model_name]
            
            # Обучим модель.
            model.fit(pd.DataFrame(X_train), y_train)
            #model.fit(X_train, y_train)
            # Сделаем предсказание.
            prediction = model.predict(X_val)

            # Создадим валидационный датасет.
            eval_df = X_val.copy()
            eval_df["target"] = y_val
            #eval_df["prediction"] = prediction

            # Сохраним результаты обучения с помощью MLFlow.
            #mlflow.evaluate(
            #    data=eval_df,
            #    targets="target",
            #    predictions="prediction",
            #    model_type="regressor",
            #    evaluators=["default"],
            #)
            # Сохраним результаты обучения с помощью MLFlow.
            signature = infer_signature(X_test, prediction)
            model_info = mlflow.sklearn.log_model(model, model_name, signature=signature)
            mlflow.evaluate(
                model=model_info.model_uri,
                data=eval_df,
                targets="target",
                #predictions="prediction",
                model_type="regressor",
                evaluators=["default"],
            )
                
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
