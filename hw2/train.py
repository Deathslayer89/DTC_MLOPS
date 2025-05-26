import os
import pickle
import click
import mlflow # Add mlflow import

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):
    mlflow.set_tracking_uri("sqlite:////home/dinesh/Programming/mlops/hw2/mlflow.db") # Add this line to set the tracking URI
    mlflow.set_experiment("nyc-taxi-experiment") # Add this line to set the experiment
    mlflow.sklearn.autolog(log_model_signatures=False, log_input_examples=False) # Disable signature and input example logging

    with mlflow.start_run(): # Wrap training code with mlflow.start_run()
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metric("rmse", rmse) # Optionally log metrics explicitly if needed, though autolog usually handles this for sklearn


if __name__ == '__main__':
    run_train()