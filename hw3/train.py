import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import mlflow
import os

def read_dataframe(filename):
    df = pd.read_parquet(filename)
    return df

def prepare_dataframe(df):
    df = df.copy()
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df

def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    df_dicts = df[categorical].to_dict(orient='records')
    return df_dicts

def main():
    # Set up MLflow with file-based tracking
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    
    # Use file-based tracking (default mlruns directory)
    mlruns_dir = os.path.join(current_dir, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    
    # Set MLflow tracking URI to use file-based storage
    mlflow.set_tracking_uri(f"file://{mlruns_dir}")
    
    # Set up experiment
    experiment_name = "nyc-taxi-experiment"
    
    # Try to get existing experiment, otherwise create new one
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Load the data
        df = read_dataframe('../data/yellow_tripdata_2023-03.parquet')
        print(f"Question 3 - Number of records loaded: {len(df)}")
        
        # Data preparation
        df = prepare_dataframe(df)
        print(f"Question 4 - Size after preparation: {len(df)}")
        
        # Prepare features
        train_dicts = prepare_dictionaries(df)
        
        # Initialize and fit DictVectorizer
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        target = 'duration'
        y_train = df[target].values
        
        # Train the model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Print model intercept
        print(f"Question 5 - Model intercept: {lr.intercept_:.2f}")
        
        # Log metrics and model with MLflow
        mlflow.log_metric("intercept", lr.intercept_)
        
        # Log the model
        mlflow.sklearn.log_model(lr, "model")
        
        # Save and log the DictVectorizer
        with open("dv.pkl", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("dv.pkl", artifact_path="preprocessor")

if __name__ == "__main__":
    main() 