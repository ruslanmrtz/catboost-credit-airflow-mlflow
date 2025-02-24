import logging
import os

import numpy as np
import pandas as pd
import mlflow
from src import utils_data
import json


def save_result(y_pred):
    sample_submission = pd.read_csv('/home/ruslan/airflow_credit/data/sample_submission.csv')
    sample_submission['TARGET'] = y_pred[:, 1]
    sample_submission.to_csv('submission.csv', index=False)


def main():
    """
    Получение тематик из текста и сохранение их в файл
    """

    with open('/home/ruslan/airflow_credit/config/models.json', 'r') as f:
        models_data = json.load(f)

    # Загрузка последних сохраненнных моделей из MLFlow
    mlflow.set_tracking_uri("http://localhost:5001")
    model_uri = f"models:/catboost_credit/{models_data['catboost_credit']}"

    model = mlflow.sklearn.load_model(model_uri)

    X_matrix = pd.read_csv('/home/ruslan/airflow_credit/data/X_test_preprocess.csv')

    y_pred = model.predict_proba(X_matrix)
    save_result(y_pred)

    print('Success predict!')
    return y_pred

if __name__ == "__main__":
    main()
