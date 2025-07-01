import pandas as pd
from datetime import datetime
import batch
from batch import prepare_data


def dt(hour: int, minute: int, second: int = 0) -> datetime:
    """Convenience wrapper used in homework instructions."""
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_processed = prepare_data(df, categorical)

    # The expected output should have 2 rows:
    # - Row 1 (duration = 9 minutes)
    # - Row 2 (duration = 8 minutes)
    # - Row 3 (duration = 0.98 minutes, which will be filtered out as < 1)
    # - Row 4 (duration = 60.02 minutes, which will be filtered out as > 60)

    assert len(df_processed) == 2  # Two rows should remain after filtering


def test_prepare_data_filters_and_casts():
    # Arrange -----------------------------------------------------------------
    data = [
        (None, None, dt(1, 1), dt(1, 10)),               # 9 minutes  -> keep
        (1, 1, dt(1, 2), dt(1, 10)),                     # 8 minutes  -> keep
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),           # <1 minute  -> drop
        (3, 4, dt(1, 2, 0), datetime(2023, 1, 2, 2, 1)), # >60 minutes -> drop
    ]
    columns = ["PULocationID", "DOLocationID", "tpep_pickup_datetime", "tpep_dropoff_datetime"]
    df_raw = pd.DataFrame(data, columns=columns)

    categorical = ["PULocationID", "DOLocationID"]

    # Act ---------------------------------------------------------------------
    df_prepared = batch.prepare_data(df_raw, categorical)

    # Assert ------------------------------------------------------------------
    assert len(df_prepared) == 2
    # Check if values are actually strings
    assert all(isinstance(x, str) for x in df_prepared['PULocationID'])
    assert all(isinstance(x, str) for x in df_prepared['DOLocationID'])
    # Check specific values
    assert df_prepared.iloc[0]['PULocationID'] == '-1'  # None should be converted to "-1"
    assert df_prepared.iloc[0]['DOLocationID'] == '-1'  # None should be converted to "-1"
    assert df_prepared.iloc[1]['PULocationID'] == '1'   # 1 should be converted to "1"
    assert df_prepared.iloc[1]['DOLocationID'] == '1'   # 1 should be converted to "1"

    # Ensure duration is within the expected range.
    assert df_prepared["duration"].between(1, 60).all()

    # Ensure categorical columns are strings with no missing values.
    for col in categorical:
        assert df_prepared[col].dtype == object  # dtype 'object' for strings in pandas
        assert df_prepared[col].isna().sum() == 0 