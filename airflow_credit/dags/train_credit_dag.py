from airflow import DAG

from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from datetime import timedelta

import train
from src import utils_data

default_args = {
    'owner': 'Ruslan Murtazin',
    'retry': 3,
    'retry_delay': timedelta(1),
    'description': 'train catboost regression'
}


dag = DAG(
    'train_catboost_credit',
    start_date=days_ago(2),
    catchup=False,
    schedule_interval='1 * * * *',
    tags=['catboost', 'credit'],
    default_args=default_args
)


def get_data():
    df = utils_data.get_data()


def preprocess_data():
    utils_data.transform_data()


def train_model():
    train.main()


get_pwd = BashOperator(task_id='get_pwd', bash_command='pwd')
get_data_operator = PythonOperator(task_id='get_data', python_callable=get_data, dag=dag)
preprocess_data_operator = PythonOperator(task_id='preprocess_data', python_callable=preprocess_data, dag=dag)
train_model_operator = PythonOperator(task_id='train_model', python_callable=train_model, dag=dag)

get_pwd >> get_data_operator >> preprocess_data_operator >> train_model_operator