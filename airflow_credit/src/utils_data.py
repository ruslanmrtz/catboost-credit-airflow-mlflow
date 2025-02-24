import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler


def get_data():
    df = pd.read_csv('/home/ruslan/airflow_credit/data/application_train.csv')

    return df


def get_test_data():
    df = pd.read_csv('/home/ruslan/airflow_credit/data/application_test.csv')

    return df


def transform_data():
    df = pd.read_csv('/home/ruslan/airflow_credit/data/application_train.csv')

    X = df.drop(['TARGET'], axis=1)
    y = df['TARGET']

    cat_features = X.describe(include='object').columns
    num_features = X.describe().columns

    cat_tranformer = Pipeline([
        ('inputer', SimpleImputer(strategy='most_frequent')),
    ])

    num_transformer = Pipeline([
        ('inputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler(with_mean=False))
    ])

    transformer = ColumnTransformer([
        ('cat', cat_tranformer, cat_features),
        ('num', num_transformer, num_features)
    ])

    joblib.dump(transformer, '/home/ruslan/airflow_credit/src/transformer.pkl')

    transformer.fit(X)

    X = pd.DataFrame(transformer.transform(X), columns=transformer.get_feature_names_out())

    X.to_csv('/home/ruslan/airflow_credit/data/X_preprocess.csv', index=False)
    y.to_csv('/home/ruslan/airflow_credit/data/y.csv', index=False)

    return X, y, cat_features


def transform_test_data():
    df = pd.read_csv('/home/ruslan/airflow_credit/data/application_test.csv')

    transformer = joblib.load('/home/ruslan/airflow_credit/src/transformer.pkl')

    X = pd.DataFrame(transformer.fit_transform(df), columns=transformer.get_feature_names_out())
    X.to_csv('/home/ruslan/airflow_credit/data/X_test_preprocess.csv', index=False)

    return X


