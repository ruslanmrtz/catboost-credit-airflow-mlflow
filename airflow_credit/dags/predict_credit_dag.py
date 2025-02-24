from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from datetime import timedelta

import predict
from src import utils_data

default_args = {
    'owner': 'Ruslan Murtazin',
    'retry': 3,
    'retry_delay': timedelta(1),
    'description': 'predict catboost regression'
}


dag = DAG(
    'predict_catboost_credit',
    start_date=days_ago(2),
    catchup=False,
    schedule_interval='*/5 * * * *',
    tags=['catboost', 'credit'],
    default_args=default_args
)


def get_data():
    df = utils_data.get_test_data()


def preprocess_data():
    utils_data.transform_test_data()


def predict_model():
    predict.main()


get_data_operator = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag)
preprocess_data_operator = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
predict_model_operator = PythonOperator(task_id='predict_model', python_callable=predict_model, dag=dag)

get_data_operator >> preprocess_data_operator >> predict_model_operator