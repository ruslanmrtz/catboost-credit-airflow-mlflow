
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
import mlflow

X, y = pd.read_csv('data/X_preprocess.csv'), pd.read_csv('data/y.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model = LinearRegression()

mlflow.set_tracking_uri("http://localhost:5001")
with mlflow.start_run(run_id='de541a030289471d8e081826a60fadbf'):
    with mlflow.start_run(run_name="Linear Regression", experiment_id='1',
                          description = "Linear Regression", nested=True):
        model.fit(X_train, y_train)

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