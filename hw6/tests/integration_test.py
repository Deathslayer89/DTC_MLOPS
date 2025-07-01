import os
import uuid
import pandas as pd
from datetime import datetime
import pytest
import sys
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

try:
    import boto3
except ImportError:
    boto3 = None

import batch

S3_ENDPOINT_URL = "http://localhost:4566"

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def test_integration():
    """Simple integration test that saves test data to S3"""
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df_input = pd.DataFrame(data, columns=columns)

    options = {
        'client_kwargs': {
            'endpoint_url': S3_ENDPOINT_URL
        }
    }

    # Set environment variables
    os.environ['S3_ENDPOINT_URL'] = S3_ENDPOINT_URL
    os.environ['INPUT_FILE_PATTERN'] = 's3://nyc-duration/in/{year:04d}-{month:02d}.parquet'
    os.environ['OUTPUT_FILE_PATTERN'] = 's3://nyc-duration/out/{year:04d}-{month:02d}.parquet'

    # Create S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=S3_ENDPOINT_URL,
        aws_access_key_id='test',
        aws_secret_access_key='test',
        region_name='us-east-1'
    )

    # Create bucket
    try:
        s3.create_bucket(Bucket='nyc-duration')
    except:
        pass

    # Save input file
    input_file = 's3://nyc-duration/in/2023-01.parquet'
    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    # Get input file size
    response = s3.head_object(Bucket='nyc-duration', Key='in/2023-01.parquet')
    file_size = response['ContentLength']
    print(f"\n Input file size: {file_size} bytes")

    # Run batch prediction
    batch.main(2023, 1)

    # Read output file
    output_file = 's3://nyc-duration/out/2023-01.parquet'
    df_output = pd.read_parquet(output_file, storage_options=options)
    duration_sum = df_output['predicted_duration'].sum()
    print(f"\n Sum of predicted durations: {duration_sum}")

if __name__ == '__main__':
    test_integration() 