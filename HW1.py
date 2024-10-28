import io
import json
import logging
import numpy as np
import os
import pandas as pd
import pickle

from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, median_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import DAG, Variable
from airflow.operators.python_operator import PythonOperator
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

# DAG для обучения Random Forest
dag_rf = DAG(
    dag_id="train_random_forest",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)

# DAG для обучения Gradient Boosting
dag_gb = DAG(
    dag_id="train_gradient_boosting",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)

# DAG для обучения Linear Regression
dag_lr = DAG(
    dag_id="train_linear_regression",
    schedule_interval="0 1 * * *",
    start_date=days_ago(2),
    catchup=False,
    tags=["mlops"],
    default_args=DEFAULT_ARGS,
)

# Функции для работы с данными
def init() -> None:
    _LOG.info("Train pipeline started.")

def get_data(model_name: str) -> None:
    housing = fetch_california_housing(as_frame=True)
    data = pd.concat([housing["data"], pd.DataFrame(housing["target"])], axis=1)

    s3_hook = S3Hook("s3_connection")
    filebuffer = io.BytesIO()
    data.to_pickle(filebuffer)
    filebuffer.seek(0)

    # Сохраняем данные в соответствующую папку модели
    s3_hook.load_file_obj(
        file_obj=filebuffer,
        key=f"ANTON_IVANOV/{model_name}/datasets/california_housing.pkl",
        bucket_name=BUCKET,
        replace=True,
    )
    _LOG.info("Data downloaded.")

def prepare_data(model_name: str) -> None:
    s3_hook = S3Hook("s3_connection")
    file = s3_hook.download_file(key=f"ANTON_IVANOV/{model_name}/datasets/california_housing.pkl", bucket_name=BUCKET)
    data = pd.read_pickle(file)

    X, y = data[FEATURES], data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_fitted = scaler.fit_transform(X_train)
    X_test_fitted = scaler.transform(X_test)

    for name, data in zip(["X_train", "X_test", "y_train", "y_test"], 
                          [X_train_fitted, X_test_fitted, y_train, y_test]):
        filebuffer = io.BytesIO()
        pickle.dump(data, filebuffer)
        filebuffer.seek(0)
        s3_hook.load_file_obj(file_obj=filebuffer, key=f"ANTON_IVANOV/{model_name}/datasets/{name}.pkl", bucket_name=BUCKET, replace=True)

    _LOG.info("Data prepared.")

# Функция для обучения модели и сохранения метрик
def train_model(model_name: str, **kwargs) -> None:
    s3_hook = S3Hook("s3_connection")
    data = {}
    for name in ["X_train", "X_test", "y_train", "y_test"]:
        file = s3_hook.download_file(key=f"ANTON_IVANOV/{model_name}/datasets/{name}.pkl", bucket_name=BUCKET)
        data[name] = pd.read_pickle(file)

    # Определяем модель
    if model_name == "random_forest":
        model = RandomForestRegressor()
    elif model_name == "gradient_boosting":
        model = GradientBoostingRegressor()
    elif model_name == "linear_regression":
        model = LinearRegression()

    model.fit(data["X_train"], data["y_train"])
    prediction = model.predict(data["X_test"])

    result = {
        "r2_score": r2_score(data["y_test"], prediction),
        "rmse": mean_squared_error(data["y_test"], prediction) ** 0.5,
        "mae": median_absolute_error(data["y_test"], prediction),
    }

    date = datetime.now().strftime("%Y_%m_%d_%H")
    filebuffer = io.BytesIO()
    filebuffer.write(json.dumps(result).encode())
    filebuffer.seek(0)
    
    # Сохраняем метрики в соответствующую папку модели
    s3_hook.load_file_obj(file_obj=filebuffer, key=f"ANTON_IVANOV/{model_name}/results/{date}.json", bucket_name=BUCKET, replace=True)

    # Передаем метрики через XCom
    return result

def save_results(**kwargs) -> None:
    metrics = kwargs['ti'].xcom_pull(task_ids='train_model')
    _LOG.info(f"Training metrics: {metrics}")
    print("Success")
    _LOG.info("Success.")

# Задачи для первого DAG (Random Forest)
task_init_rf = PythonOperator(task_id="init", python_callable=init, dag=dag_rf)

task_download_data_rf = PythonOperator(
    task_id="download_data", python_callable=get_data, op_kwargs={'model_name': 'random_forest'}, dag=dag_rf
)

task_prepare_data_rf = PythonOperator(
    task_id="data_preparation", python_callable=prepare_data, op_kwargs={'model_name': 'random_forest'}, dag=dag_rf
)

task_train_model_rf = PythonOperator(
    task_id="train_model", python_callable=train_model,
    op_kwargs={'model_name': 'random_forest'},  # Передаем имя модели
    dag=dag_rf
)

task_save_results_rf = PythonOperator(
    task_id="save_results", python_callable=save_results,
    provide_context=True,  # Позволяем передавать контекст
    dag=dag_rf
)

task_init_rf >> task_download_data_rf >> task_prepare_data_rf >> task_train_model_rf >> task_save_results_rf

# Задачи для второго DAG (Gradient Boosting)
task_init_gb = PythonOperator(task_id="init", python_callable=init, dag=dag_gb)

task_download_data_gb = PythonOperator(
    task_id="download_data", python_callable=get_data, op_kwargs={'model_name': 'gradient_boosting'}, dag=dag_gb
)

task_prepare_data_gb = PythonOperator(
    task_id="data_preparation", python_callable=prepare_data, op_kwargs={'model_name': 'gradient_boosting'}, dag=dag_gb
)

task_train_model_gb = PythonOperator(
    task_id="train_model", python_callable=train_model,
    op_kwargs={'model_name': 'gradient_boosting'},  # Передаем имя модели
    dag=dag_gb
)

task_save_results_gb = PythonOperator(
    task_id="save_results", python_callable=save_results,
    provide_context=True,  # Позволяем передавать контекст
    dag=dag_gb
)

task_init_gb >> task_download_data_gb >> task_prepare_data_gb >> task_train_model_gb >> task_save_results_gb

# Задачи для третьего DAG (Linear Regression)
task_init_lr = PythonOperator(task_id="init", python_callable=init, dag=dag_lr)

task_download_data_lr = PythonOperator(
    task_id="download_data", python_callable=get_data, op_kwargs={'model_name': 'linear_regression'}, dag=dag_lr
)

task_prepare_data_lr = PythonOperator(
    task_id="data_preparation", python_callable=prepare_data, op_kwargs={'model_name': 'linear_regression'}, dag=dag_lr
)

task_train_model_lr = PythonOperator(
    task_id="train_model", python_callable=train_model,
    op_kwargs={'model_name': 'linear_regression'},  # Передаем имя модели
    dag=dag_lr
)

task_save_results_lr = PythonOperator(
    task_id="save_results", python_callable=save_results,
    provide_context=True,  # Позволяем передавать контекст
    dag=dag_lr
)

task_init_lr >> task_download_data_lr >> task_prepare_data_lr >> task_train_model_lr >> task_save_results_lr
