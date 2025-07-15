"""
Test suite for data pipeline functionality.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
from src.data.data_loader import load_raw_data, clean_data, validate_data, save_processed_data
from src.features.feature_engineering import (
    create_time_features, create_lag_features, create_rolling_features,
    create_interaction_features, engineer_features, select_features, split_data
)


class TestDataLoader:
    """Test cases for data loading functionality."""
    
    def test_load_raw_data_success(self):
        """Test successful data loading."""
        # Create sample data file
        sample_data = """Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000
16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(sample_data)
            temp_file = f.name
        
        try:
            df = load_raw_data(temp_file)
            
            # Assertions
            assert len(df) == 2
            assert 'datetime' in df.columns
            assert 'Global_active_power' in df.columns
            assert pd.api.types.is_datetime64_any_dtype(df['datetime'])
            
        finally:
            os.unlink(temp_file)
    
    def test_load_raw_data_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("non_existent_file.txt")
    
    def test_clean_data_handles_missing_values(self):
        """Test that clean_data properly handles missing values."""
        # Create DataFrame with missing values
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=3, freq='H'),
            'Global_active_power': ['1.0', '?', '2.0'],
            'Global_reactive_power': ['0.1', '0.2', '?'],
            'Voltage': ['230', '?', '235'],
            'Global_intensity': ['4.0', '5.0', '6.0'],
            'Sub_metering_1': [0.0, 1.0, 2.0],
            'Sub_metering_2': [1.0, 2.0, 3.0],
            'Sub_metering_3': [17.0, 18.0, 19.0]
        })
        
        cleaned_df = clean_data(df)
        
        # Assertions
        assert cleaned_df['Global_active_power'].isna().sum() == 1
        assert cleaned_df['Global_reactive_power'].isna().sum() == 1
        assert cleaned_df['Voltage'].isna().sum() == 1
        assert 'Total_sub_metering' in cleaned_df.columns
        assert len(cleaned_df) == 2  # One row removed due to missing target
    
    def test_clean_data_creates_total_submetering(self):
        """Test that clean_data creates total sub-metering feature."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=2, freq='H'),
            'Global_active_power': [1.0, 2.0],
            'Global_reactive_power': [0.1, 0.2],
            'Voltage': [230, 235],
            'Global_intensity': [4.0, 5.0],
            'Sub_metering_1': [1.0, 2.0],
            'Sub_metering_2': [2.0, 3.0],
            'Sub_metering_3': [3.0, 4.0]
        })
        
        cleaned_df = clean_data(df)
        
        # Assertions
        assert 'Total_sub_metering' in cleaned_df.columns
        assert cleaned_df['Total_sub_metering'].iloc[0] == 6.0
        assert cleaned_df['Total_sub_metering'].iloc[1] == 9.0
    
    def test_validate_data_success(self):
        """Test successful data validation."""
        # Create valid DataFrame
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=2, freq='H'),
            'Global_active_power': [1.0, 2.0],
            'Global_reactive_power': [0.1, 0.2],
            'Voltage': [230, 235],
            'Global_intensity': [4.0, 5.0],
            'Sub_metering_1': [1.0, 2.0],
            'Sub_metering_2': [2.0, 3.0],
            'Sub_metering_3': [3.0, 4.0]
        })
        
        result = validate_data(df)
        assert result is True
    
    def test_validate_data_missing_columns(self):
        """Test validation failure for missing columns."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=2, freq='H'),
            'Global_active_power': [1.0, 2.0]
        })
        
        result = validate_data(df)
        assert result is False
    
    def test_save_processed_data(self):
        """Test saving processed data."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=2, freq='H'),
            'Global_active_power': [1.0, 2.0],
            'feature1': [10, 20]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            temp_file = f.name
        
        try:
            save_processed_data(df, temp_file)
            
            # Load and verify
            loaded_df = pd.read_csv(temp_file)
            assert len(loaded_df) == 2
            assert 'Global_active_power' in loaded_df.columns
            assert 'feature1' in loaded_df.columns
            
        finally:
            os.unlink(temp_file)


class TestFeatureEngineering:
    """Test cases for feature engineering functionality."""
    
    def test_create_time_features(self):
        """Test time feature creation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=24, freq='H'),
            'Global_active_power': np.random.rand(24)
        })
        
        df_with_features = create_time_features(df)
        
        # Assertions
        assert 'hour' in df_with_features.columns
        assert 'day_of_week' in df_with_features.columns
        assert 'month' in df_with_features.columns
        assert 'season' in df_with_features.columns
        assert 'is_weekend' in df_with_features.columns
        assert 'is_working_hours' in df_with_features.columns
        assert 'time_of_day' in df_with_features.columns
        
        # Check value ranges
        assert df_with_features['hour'].min() >= 0
        assert df_with_features['hour'].max() <= 23
        assert df_with_features['day_of_week'].min() >= 0
        assert df_with_features['day_of_week'].max() <= 6
        assert df_with_features['month'].min() >= 1
        assert df_with_features['month'].max() <= 12
    
    def test_create_lag_features(self):
        """Test lag feature creation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='H'),
            'Global_active_power': np.random.rand(200)
        })
        
        df_with_lags = create_lag_features(df, lags=[1, 24])
        
        # Assertions
        assert 'Global_active_power_lag_1' in df_with_lags.columns
        assert 'Global_active_power_lag_24' in df_with_lags.columns
        assert 'Voltage_lag_1' in df_with_lags.columns
        assert 'Voltage_lag_24' in df_with_lags.columns
        
        # Check lag values
        assert df_with_lags['Global_active_power_lag_1'].iloc[1] == df_with_lags['Global_active_power'].iloc[0]
        assert pd.isna(df_with_lags['Global_active_power_lag_1'].iloc[0])
    
    def test_create_rolling_features(self):
        """Test rolling feature creation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='H'),
            'Global_active_power': np.random.rand(100)
        })
        
        df_with_rolling = create_rolling_features(df, windows=[24])
        
        # Assertions
        assert 'Global_active_power_rolling_mean_24' in df_with_rolling.columns
        assert 'Global_active_power_rolling_std_24' in df_with_rolling.columns
        assert 'Global_active_power_rolling_min_24' in df_with_rolling.columns
        assert 'Global_active_power_rolling_max_24' in df_with_rolling.columns
        
        # Check rolling values
        assert not pd.isna(df_with_rolling['Global_active_power_rolling_mean_24'].iloc[0])
    
    def test_create_interaction_features(self):
        """Test interaction feature creation."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='H'),
            'Global_active_power': np.random.rand(10),
            'Global_intensity': np.random.rand(10) + 1,  # Avoid zero
            'Voltage': np.random.rand(10) * 10 + 220,
            'Sub_metering_1': np.random.rand(10),
            'Sub_metering_2': np.random.rand(10),
            'Sub_metering_3': np.random.rand(10),
            'hour': np.random.randint(0, 24, 10),
            'day_of_week': np.random.randint(0, 7, 10)
        })
        
        df_with_interactions = create_interaction_features(df)
        
        # Assertions
        assert 'power_efficiency' in df_with_interactions.columns
        assert 'kitchen_ratio' in df_with_interactions.columns
        assert 'laundry_ratio' in df_with_interactions.columns
        assert 'hvac_ratio' in df_with_interactions.columns
        assert 'is_peak_hour' in df_with_interactions.columns
        
        # Check ratio values sum to 1 (approximately)
        total_ratio = (df_with_interactions['kitchen_ratio'] + 
                      df_with_interactions['laundry_ratio'] + 
                      df_with_interactions['hvac_ratio'])
        assert all(abs(total_ratio - 1.0) < 0.1)  # Allow for small numerical errors
    
    def test_select_features(self):
        """Test feature selection."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='H'),
            'Global_active_power': np.random.rand(10),
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'time_of_day': ['morning'] * 5 + ['evening'] * 5
        })
        
        X, y = select_features(df, target_col='Global_active_power')
        
        # Assertions
        assert len(X) == 10
        assert len(y) == 10
        assert 'Global_active_power' not in X.columns
        assert 'datetime' not in X.columns
        assert 'feature1' in X.columns
        assert 'feature2' in X.columns
        
        # Check one-hot encoding
        assert 'time_of_day_evening' in X.columns
        assert 'time_of_day_morning' in X.columns
    
    def test_split_data(self):
        """Test data splitting."""
        X = pd.DataFrame(np.random.rand(100, 5))
        y = pd.Series(np.random.rand(100))
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, val_size=0.2)
        
        # Assertions
        assert len(X_train) == 64  # 100 * 0.8 * 0.8
        assert len(X_val) == 16    # 100 * 0.8 * 0.2
        assert len(X_test) == 20   # 100 * 0.2
        assert len(y_train) == 64
        assert len(y_val) == 16
        assert len(y_test) == 20
        
        # Check no overlap
        assert len(set(X_train.index) & set(X_val.index)) == 0
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0
    
    def test_engineer_features_integration(self):
        """Test complete feature engineering pipeline."""
        df = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=200, freq='H'),
            'Global_active_power': np.random.rand(200),
            'Global_reactive_power': np.random.rand(200),
            'Voltage': np.random.rand(200) * 10 + 220,
            'Global_intensity': np.random.rand(200) + 1,
            'Sub_metering_1': np.random.rand(200),
            'Sub_metering_2': np.random.rand(200),
            'Sub_metering_3': np.random.rand(200)
        })
        
        df_features = engineer_features(df)
        
        # Assertions
        assert len(df_features) < len(df)  # Some rows removed due to NaN
        assert 'hour' in df_features.columns
        assert 'Global_active_power_lag_1' in df_features.columns
        assert 'Global_active_power_rolling_mean_24' in df_features.columns
        assert 'power_efficiency' in df_features.columns
        
        # Check no NaN values
        assert df_features.isna().sum().sum() == 0


class TestDataFixtures:
    """Test fixtures and utilities."""
    
    @pytest.fixture
    def sample_energy_data(self):
        """Create sample energy consumption data."""
        return pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='H'),
            'Global_active_power': np.random.rand(100) * 5,
            'Global_reactive_power': np.random.rand(100) * 0.5,
            'Voltage': np.random.rand(100) * 10 + 220,
            'Global_intensity': np.random.rand(100) * 10 + 1,
            'Sub_metering_1': np.random.rand(100) * 10,
            'Sub_metering_2': np.random.rand(100) * 10,
            'Sub_metering_3': np.random.rand(100) * 10
        })
    
    def test_sample_data_fixture(self, sample_energy_data):
        """Test sample data fixture."""
        assert len(sample_energy_data) == 100
        assert 'datetime' in sample_energy_data.columns
        assert 'Global_active_power' in sample_energy_data.columns
        assert sample_energy_data['Global_active_power'].min() >= 0
        assert sample_energy_data['Global_active_power'].max() <= 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
