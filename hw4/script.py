import argparse
import pickle
import pandas as pd
import requests
import os

# Define categorical columns
categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def main(year, month):
    # Download the file
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    filename = f'yellow_tripdata_{year:04d}-{month:02d}.parquet'

    if not os.path.exists(filename):
        response = requests.get(url)
        with open(filename, 'wb') as f:
            f.write(response.content)

    # Load the model
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    # Read the data
    df = read_data(filename)

    # Prepare features
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)

    # Make predictions
    y_pred = model.predict(X_val)

    # Calculate mean predicted duration
    mean_pred = y_pred.mean()
    print(f"Mean predicted duration: {mean_pred:.2f}")

    # Create output dataframe
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    # Save as parquet
    output_file = f'output_{year:04d}_{month:02d}.parquet'
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--month', type=int, required=True)
    args = parser.parse_args()

    main(args.year, args.month)