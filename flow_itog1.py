import mlflow
import os
import io
import json
import pickle
import pandas as pd
import logging

from airflow.models import DAG, Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.operators.python_operator import PythonOperator
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.datasets import fetch_california_housing
from datetime import datetime, timedelta
from typing import Any, Dict, Literal
from airflow.utils.dates import days_ago

_LOG = logging.getLogger()
_LOG.addHandler(logging.StreamHandler())

BUCKET = Variable.get("S3_BUCKET")
FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
TARGET = "MedHouseVal"
DEFAULT_ARGS = {
    "owner": "ANTON_IVANOV",
    "retry": 3,
    "retry_delay": timedelta(minutes=1)
}
dag = DAG(
    dag_id="anton_ivanov_1",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)

model_names = ["random_forest", "linear_regression", "gradient_boosting"]
models = dict(
    zip(model_names, [
        RandomForestRegressor(),
        LinearRegression(),
        GradientBoostingRegressor()
    ]))


def configure_mlflow():
    
    for key in [
        "MLFLOW_TRACKING_URI",
        "AWS_ENDPOINT_URL",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ[key] = Variable.get(key)
    _LOG.info(os.getenv('MLFLOW_TRACKING_URI', 'NO_URI'))

def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
   configure_mlflow()
   experiment_id = mlflow.set_experiment("anton_ivanov").experiment_id
   with mlflow.start_run() as run:
       run_id = run.info.run_id
   return {"experiment_id": experiment_id, "run_id": run_id}
# def init(m_name: Literal["random_forest", "linear_regression", "desicion_tree"]) -> Dict[str, Any]:
    # configure_mlflow()
    # mlflow.search_experiments()

def get_data(**kwargs) -> Dict[str, Any]:
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"], columns=[TARGET])], axis=1)

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"ANTON_IVANOV/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    return {}


def prepare_data(**kwargs) -> Dict[str, Any]:
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"ANTON_IVANOV/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    for name, dataset in zip(["X_train", "X_test", "y_train", "y_test"],
                             [X_train_fitted, X_test_fitted, y_train, y_test]):
        filebuffer = io.BytesIO()
        pickle.dump(dataset, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(file_obj=filebuffer, key=f"ANTON_IVANOV/datasets/{name}.pkl", bucket_name=BUCKET, replace=True)

    return {}


def train_model(model_name: str, **kwargs) -> Dict[str, Any]:
    #Получаем experiment_id и run_id из предыдущих шагов
    metrics = kwargs["ti"].xcom_pull(task_ids="init")
    experiment_id = metrics["experiment_id"]
    run_id = metrics["run_id"]
    
    #Загружаем подготовленные данные
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"ANTON_IVANOV/datasets/X_train.pkl", bucket_name=BUCKET)
    X_train = pickle.load(file)
    file = s3_hook.download_file(key=f"ANTON_IVANOV/datasets/y_train.pkl", bucket_name=BUCKET)
    y_train = pickle.load(file)
    file = s3_hook.download_file(key=f"ANTON_IVANOV/datasets/X_test.pkl", bucket_name=BUCKET)
    X_test = pickle.load(file)
    file = s3_hook.download_file(key=f"ANTON_IVANOV/datasets/y_test.pkl", bucket_name=BUCKET)
    y_test = pickle.load(file)

    #Загружаем модель
    model = models[model_name]
    
    #Логируем модель в MLflow
    with mlflow.start_run(experiment_id=experiment_id, run_name=model_name) as run:
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, model_name)
        
        #Прогнозируем и оцениваем модель
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
    
    #Сохраняем модель в S3
    model_path = f"ANTON_IVANOV/results/{model_name}/model.pkl"
    filebuffer = io.BytesIO()
    pickle.dump(model, filebuffer)
    filebuffer.seek(0)
    
    s3_hook.load_file_obj(file_obj=filebuffer, key=model_path, bucket_name=BUCKET, replace=True)
    return {}


def save_results(**kwargs) -> None:
    s3_hook = S3Hook("s3_connection")
    #Предполагается, что результаты уже сохранены на S3 во время тренировки модели
    

#Определяем таски
task_init = PythonOperator(
    task_id="init",
    python_callable=init,
    op_kwargs={"m_name": "random_forest"},  # или любой другой список моделей
    provide_context=True,
    dag=dag,
)

task_get_data = PythonOperator(
    task_id="get_data",
    python_callable=get_data,
    provide_context=True,
    dag=dag,
)

task_prepare_data = PythonOperator(
    task_id="prepare_data",
    python_callable=prepare_data,
    provide_context=True,
    dag=dag,
)

#Задаем параллельное обучение моделей
training_model_tasks = [
    PythonOperator(
        task_id=f"train_model_{model_name}",
        python_callable=train_model,
        op_kwargs={"model_name": model_name},
        provide_context=True,
        dag=dag,
    ) for model_name in model_names
]

task_save_results = PythonOperator(
    task_id="save_results",
    python_callable=save_results,
    provide_context=True,
    dag=dag,
)

# Устанавливаем зависимости задач
task_init >> task_get_data >> task_prepare_data >> training_model_tasks >> task_save_results
