from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import joblib

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from datetime import timedelta
from typing import Dict
import mlflow


default_args = {
    'owner': 'Ruslan Murtazin',
    'retry': 3,
    'retry_delay': timedelta(minutes=1),
    'description': 'regression 3 models: linear regression, random forest and histgb'
}


dag = DAG(
    'train_3_model_mlflow',
    start_date=days_ago(2),
    catchup=False,
    schedule='1 * * * *',
    tags=['regression', 'training', 'mlflow'],
    default_args=default_args
)


def get_data_from_sklearn():
    data = fetch_california_housing(as_frame=True)
    X = data.data
    y = data.target
    X.to_csv('data/X.csv', index=False)
    y.to_csv('data/y.csv', index=False)


def prepare_data() -> Dict[str, str]:
    metrics = {}
    X = pd.read_csv('data/X.csv')

    pipe = Pipeline(
            [
                ('scaler', StandardScaler())
        ]
    )
    X = pd.DataFrame(pipe.fit_transform(X), columns=X.columns)
    X.to_csv('data/X_preprocess.csv', index=False)

    mlflow.set_tracking_uri("http://localhost:5001")
    try:
        mlflow.create_experiment('3_sklearn_models_regression')
    except Exception as e:
        print(e)
        print('Эксперимент уже создан!')
    exp = mlflow.set_experiment('3_sklearn_models_regression')
    metrics['exp_id'] = exp.experiment_id
    print(metrics['exp_id'])

    run_id = mlflow.start_run(experiment_id=exp.experiment_id, run_name='train_models').__dict__['_info'].run_id
    metrics['run_id'] = run_id
    return metrics


def train_rf(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull('prepare_data')
    print(metrics['exp_id'])

    X, y = pd.read_csv('data/X_preprocess.csv'), pd.read_csv('data/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = RandomForestRegressor()

    mlflow.set_tracking_uri("http://localhost:5001")
    with mlflow.start_run(run_id=metrics["run_id"], experiment_id=metrics['exp_id']):
        with mlflow.start_run(run_name="Random Forest", experiment_id=metrics['exp_id'],
                              description="Random Forest", nested=True):
            try:
                model.fit(X_train, y_train)
            except Exception as e:
                print(f"Error during model fitting: {e}")
                raise

            eval_df = X_test.copy()
            eval_df['prediction'] = model.predict(X_test)
            eval_df['target'] = y_test

            mlflow.sklearn.log_model(model,
                                     artifact_path='mlflow',
                                     registered_model_name='Random Forest')

            mlflow.evaluate(data=eval_df,
                            predictions='prediction',
                            targets='target',
                            model_type='regressor',
                            evaluators=['default'])

            mlflow.log_params(model.get_params())


def train_lr(**kwargs):

    ti = kwargs['ti']
    metrics = ti.xcom_pull('prepare_data')

    X, y = pd.read_csv('data/X_preprocess.csv'), pd.read_csv('data/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LinearRegression()

    mlflow.set_tracking_uri("http://localhost:5001")
    with mlflow.start_run(run_id=metrics["run_id"], run_name="Linear Regression",
                          description = "Linear Regression", nested=True):
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

        eval_df = X_test.copy()
        eval_df['prediction'] = model.predict(X_test)
        eval_df['target'] = y_test

        mlflow.sklearn.log_model(model,
                                 artifact_path='mlflow',
                                 registered_model_name='Linear Regression')

        mlflow.evaluate(data=eval_df,
                        predictions='prediction',
                        targets='target',
                        model_type='regressor',
                        evaluators=['default'])

        mlflow.log_params(model.get_params())


def train_hgb(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull('prepare_data')

    X, y = pd.read_csv('data/X_preprocess.csv'), pd.read_csv('data/y.csv')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = LinearRegression()

    mlflow.set_tracking_uri("http://localhost:5001")
    with mlflow.start_run(run_id=metrics["run_id"], run_name="HistGB",
                          description="HistGB", nested=True):
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error during model fitting: {e}")
            raise

        eval_df = X_test.copy()
        eval_df['prediction'] = model.predict(X_test)
        eval_df['target'] = y_test

        mlflow.sklearn.log_model(model,
                                 artifact_path='mlflow',
                                 registered_model_name='HistGB')

        mlflow.evaluate(data=eval_df,
                        predictions='prediction',
                        targets='target',
                        model_type='regressor',
                        evaluators=['default'])

        mlflow.log_params(model.get_params())


def save_results(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull('prepare_data')

    joblib.dump(metrics, f'mlflow/{metrics["exp_id"]}/{metrics["run_id"]}/info_mlops.json')

    print('Success!!!')


init_operator = BashOperator(task_id='init', bash_command='echo $AIRFLOW_HOME', dag=dag)
get_data_operator = PythonOperator(task_id='get_data_from_sklearn', python_callable=get_data_from_sklearn, dag=dag)
prepare_data_operator = PythonOperator(task_id='prepare_data', python_callable=prepare_data, dag=dag, provide_context=True)
train_rf_operator = PythonOperator(task_id='train_rf', python_callable=train_rf, dag=dag, provide_context=True)
# train_lr_operator = PythonOperator(task_id='train_lr', python_callable=train_lr, dag=dag, provide_context=True)
# train_hgb_operator = PythonOperator(task_id='train_hgb', python_callable=train_hgb, dag=dag, provide_context=True)
save_result_operator = PythonOperator(task_id='save_results', python_callable=save_results, dag=dag, provide_context=True)

var = init_operator >> get_data_operator >> prepare_data_operator >> train_rf_operator >> save_result_operator


