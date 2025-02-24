import logging
import os
import json
import pandas as pd

from mlflow.tracking import MlflowClient

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score

import mlflow
from src import utils_data


def get_version_model(config_name, client):
    """
    Получение последней версии модели из MLFlow
    """
    dict_push = {}
    for count, value in enumerate(client.search_model_versions(f"name='{config_name}'")):
        # Все версии модели
        dict_push[count] = value
    return dict_push

def main():
    """
    Получение данных и обучение, сохранение модели
    """
    X, y = pd.read_csv('/home/ruslan/airflow_credit/data/X_preprocess.csv'), pd.read_csv('/home/ruslan/airflow_credit/data/y.csv')
    cat_features = [col for col in X.columns if col not in X.describe().columns]

    # Обучение линейной модели на поиска сформированных тематик
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        random_state=1)
    model = CatBoostClassifier(verbose=1000, cat_features=cat_features, task_type="GPU", loss_function='Logloss')
    print(y_train)

    mlflow.create_experiment(
        name='CreditNew761',
        artifact_location="mlflow"
    )
    # MLFlow трэкинг
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment('CreditNew4')
    with mlflow.start_run():
        model.fit(X_train, y_train)

        # Логирование модели и параметров
        mlflow.log_param('f1',
                         f1_score(y_test, model.predict(X_test)))
        mlflow.log_param('accuracy',
                         accuracy_score(y_test, model.predict(X_test)))
        mlflow.log_param('precision',
                         precision_score(y_test, model.predict(X_test)))

        mlflow.sklearn.log_model(model,
                                 artifact_path="mlflow",
                                 registered_model_name=f"catboost_credit")

        mlflow.log_artifact(local_path='/home/ruslan/airflow_credit/train.py',
                            artifact_path='code')
        mlflow.end_run()

    # Получение последней версии модели и сохраннение в файлы
    client = MlflowClient()
    last_version = get_version_model('catboost_credit', client)[0].version
    with open('/home/ruslan/airflow_credit/config/models.json', 'r') as f:
        models_data = json.load(f)
    models_data['catboost_credit'] = last_version
    with open('/home/ruslan/airflow_credit/config/models.json', 'w') as f:
        json.dump(models_data, f)

    print('Success')




if __name__ == "__main__":
    main()
