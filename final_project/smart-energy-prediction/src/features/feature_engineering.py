"""
Feature engineering utilities for Smart Energy Prediction project.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from datetime column.
    
    Args:
        df: DataFrame with datetime column
        
    Returns:
        DataFrame with additional time features
    """
    logger.info("Creating time-based features...")
    
    df_features = df.copy()
    
    # Extract time components
    df_features['hour'] = df_features['datetime'].dt.hour
    df_features['day_of_week'] = df_features['datetime'].dt.dayofweek
    df_features['month'] = df_features['datetime'].dt.month
    df_features['day_of_month'] = df_features['datetime'].dt.day
    df_features['quarter'] = df_features['datetime'].dt.quarter
    df_features['year'] = df_features['datetime'].dt.year
    
    # Create season feature
    season_mapping = {
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    }
    df_features['season'] = df_features['month'].map(season_mapping)
    
    # Create time of day categories
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    df_features['time_of_day'] = df_features['hour'].apply(categorize_time)
    
    # Create weekend feature
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Create working hours feature
    df_features['is_working_hours'] = (
        (df_features['hour'] >= 9) & 
        (df_features['hour'] <= 17) & 
        (df_features['day_of_week'] < 5)
    ).astype(int)
    
    logger.info(f"Created {len([col for col in df_features.columns if col not in df.columns])} time-based features")
    
    return df_features


def create_lag_features(df: pd.DataFrame, 
                       target_col: str = 'Global_active_power',
                       lags: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """
    Create lag features for time series.
    
    Args:
        df: DataFrame with time series data
        target_col: Column to create lags for
        lags: List of lag periods (in minutes)
        
    Returns:
        DataFrame with lag features
    """
    logger.info(f"Creating lag features for {target_col} with lags: {lags}")
    
    df_lags = df.copy()
    
    # Sort by datetime to ensure proper ordering
    df_lags = df_lags.sort_values('datetime')
    
    # Create lag features
    for lag in lags:
        lag_col_name = f'{target_col}_lag_{lag}'
        df_lags[lag_col_name] = df_lags[target_col].shift(lag)
        
        # Create lag feature for other important columns
        if lag <= 24:  # Only for short-term lags
            df_lags[f'Voltage_lag_{lag}'] = df_lags['Voltage'].shift(lag)
            df_lags[f'Global_intensity_lag_{lag}'] = df_lags['Global_intensity'].shift(lag)
    
    logger.info(f"Created lag features with {len(lags)} different lag periods")
    
    return df_lags


def create_rolling_features(df: pd.DataFrame, 
                           target_col: str = 'Global_active_power',
                           windows: List[int] = [24, 168, 720]) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: DataFrame with time series data
        target_col: Column to create rolling features for
        windows: List of window sizes (in minutes)
        
    Returns:
        DataFrame with rolling features
    """
    logger.info(f"Creating rolling features for {target_col} with windows: {windows}")
    
    df_rolling = df.copy()
    
    # Sort by datetime
    df_rolling = df_rolling.sort_values('datetime')
    
    # Create rolling features
    for window in windows:
        # Rolling mean
        df_rolling[f'{target_col}_rolling_mean_{window}'] = (
            df_rolling[target_col].rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std
        df_rolling[f'{target_col}_rolling_std_{window}'] = (
            df_rolling[target_col].rolling(window=window, min_periods=1).std()
        )
        
        # Rolling min/max
        df_rolling[f'{target_col}_rolling_min_{window}'] = (
            df_rolling[target_col].rolling(window=window, min_periods=1).min()
        )
        
        df_rolling[f'{target_col}_rolling_max_{window}'] = (
            df_rolling[target_col].rolling(window=window, min_periods=1).max()
        )
    
    logger.info(f"Created rolling features with {len(windows)} different window sizes")
    
    return df_rolling


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features between variables.
    
    Args:
        df: DataFrame with features
        
    Returns:
        DataFrame with interaction features
    """
    logger.info("Creating interaction features...")
    
    df_interactions = df.copy()
    
    # Power efficiency features (with safe division)
    df_interactions['power_efficiency'] = np.where(
        df_interactions['Global_intensity'] > 0,
        df_interactions['Global_active_power'] / df_interactions['Global_intensity'],
        0
    )
    
    # Voltage stability (with safe division)
    voltage_rolling_std = df_interactions['Voltage'].rolling(window=60, min_periods=1).std()
    df_interactions['voltage_stability'] = np.where(
        voltage_rolling_std > 0,
        df_interactions['Voltage'] / voltage_rolling_std,
        df_interactions['Voltage']
    )
    
    # Sub-metering ratios (with safe division)
    total_submetering = (
        df_interactions['Sub_metering_1'] + 
        df_interactions['Sub_metering_2'] + 
        df_interactions['Sub_metering_3']
    )
    
    df_interactions['kitchen_ratio'] = np.where(
        total_submetering > 0,
        df_interactions['Sub_metering_1'] / total_submetering,
        0
    )
    
    df_interactions['laundry_ratio'] = np.where(
        total_submetering > 0,
        df_interactions['Sub_metering_2'] / total_submetering,
        0
    )
    
    df_interactions['hvac_ratio'] = np.where(
        total_submetering > 0,
        df_interactions['Sub_metering_3'] / total_submetering,
        0
    )
    
    # Peak usage indicator
    df_interactions['is_peak_hour'] = (
        (df_interactions['hour'].isin([8, 9, 18, 19, 20])) & 
        (df_interactions['day_of_week'] < 5)
    ).astype(int)
    
    logger.info("Created interaction features")
    
    return df_interactions


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        DataFrame with all engineered features
    """
    logger.info("Starting feature engineering pipeline...")
    
    # Apply all feature engineering steps
    df_features = create_time_features(df)
    df_features = create_lag_features(df_features)
    df_features = create_rolling_features(df_features)
    df_features = create_interaction_features(df_features)
    
    # Clean data to handle infinity and NaN values
    df_features = clean_data(df_features)
    
    # Remove rows with NaN values created by lag/rolling features
    initial_len = len(df_features)
    df_features = df_features.dropna()
    final_len = len(df_features)
    
    logger.info(f"Removed {initial_len - final_len} rows with NaN values after feature engineering")
    logger.info(f"Final feature set shape: {df_features.shape}")
    logger.info("Feature engineering completed")
    
    return df_features


def select_features(df: pd.DataFrame, 
                   target_col: str = 'Global_active_power',
                   exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select features for modeling.
    
    Args:
        df: DataFrame with all features
        target_col: Target column name
        exclude_cols: Columns to exclude from features
        
    Returns:
        Tuple of (features_df, target_series)
    """
    logger.info("Selecting features for modeling...")
    
    if exclude_cols is None:
        exclude_cols = ['datetime']
    
    # Define feature columns
    feature_cols = [col for col in df.columns 
                   if col not in [target_col] + exclude_cols]
    
    # Handle categorical variables
    categorical_cols = ['time_of_day']
    for col in categorical_cols:
        if col in feature_cols:
            # One-hot encode categorical variables
            df_encoded = pd.get_dummies(df[col], prefix=col)
            df = pd.concat([df, df_encoded], axis=1)
            feature_cols.remove(col)
            feature_cols.extend(df_encoded.columns.tolist())
    
    X = df[feature_cols]
    y = df[target_col]
    
    logger.info(f"Selected {len(feature_cols)} features for modeling")
    logger.info(f"Target variable: {target_col}")
    
    return X, y


def split_data(X: pd.DataFrame, 
               y: pd.Series,
               test_size: float = 0.2,
               val_size: float = 0.2,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                               pd.Series, pd.Series, pd.Series]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Features
        y: Target
        test_size: Test set size
        val_size: Validation set size
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Splitting data into train/validation/test sets...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data by handling infinity and NaN values.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data - handling infinity and NaN values...")
    
    df_clean = df.copy()
    
    # Replace infinity values with NaN
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
    
    # Log the number of problematic values
    inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
    nan_count = df_clean.isnull().sum().sum()
    
    if inf_count > 0:
        logger.warning(f"Found {inf_count} infinity values, replaced with NaN")
    if nan_count > 0:
        logger.warning(f"Found {nan_count} NaN values after cleaning")
    
    # Fill NaN values with forward fill, then backward fill, then 0
    df_clean = df_clean.ffill().bfill().fillna(0)
    
    logger.info("Data cleaning completed")
    
    return df_clean


def scale_features(X_train: pd.DataFrame, 
                  X_val: pd.DataFrame, 
                  X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """
    Scale features using StandardScaler.
    
    Args:
        X_train: Training features
        X_val: Validation features
        X_test: Test features
        
    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler)
    """
    logger.info("Scaling features...")
    
    # Clean data before scaling
    X_train_clean = clean_data(X_train)
    X_val_clean = clean_data(X_val)
    X_test_clean = clean_data(X_test)
    
    scaler = StandardScaler()
    
    # Fit on training data only
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train_clean),
        columns=X_train_clean.columns,
        index=X_train_clean.index
    )
    
    # Transform validation and test sets
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val_clean),
        columns=X_val_clean.columns,
        index=X_val_clean.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test_clean),
        columns=X_test_clean.columns,
        index=X_test_clean.index
    )
    
    logger.info("Feature scaling completed")
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


if __name__ == "__main__":
    # Example usage
    try:
        import joblib
        import os
        
        # Load processed data
        df = pd.read_csv("data/processed/processed_energy_data.csv")
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Engineer features
        df_features = engineer_features(df)
        
        # Select features
        X, y = select_features(df_features)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
        
        # Save processed data for model training
        os.makedirs("data/processed", exist_ok=True)
        
        joblib.dump(X_train_scaled, "data/processed/X_train_scaled.pkl")
        joblib.dump(X_val_scaled, "data/processed/X_val_scaled.pkl")
        joblib.dump(X_test_scaled, "data/processed/X_test_scaled.pkl")
        joblib.dump(y_train, "data/processed/y_train.pkl")
        joblib.dump(y_val, "data/processed/y_val.pkl")
        joblib.dump(y_test, "data/processed/y_test.pkl")
        joblib.dump(scaler, "data/processed/scaler.pkl")
        
        logger.info("Processed data saved successfully!")
        logger.info("Feature engineering pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        raise
