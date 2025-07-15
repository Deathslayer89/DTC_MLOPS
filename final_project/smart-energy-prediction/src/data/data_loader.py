"""
Data loading and downloading utilities for Smart Energy Prediction project.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import requests
import zipfile
import logging
from typing import Optional
from urllib.parse import urlparse
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_data(url: str = None, data_dir: str = "data/raw") -> str:
    """
    Download UCI Household Power Consumption dataset.
    
    Args:
        url: URL to download from (optional)
        data_dir: Directory to save data
        
    Returns:
        Path to downloaded file
    """
    if url is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    
    # Create directory if it doesn't exist
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    # Download file
    zip_path = os.path.join(data_dir, "household_power_consumption.zip")
    txt_path = os.path.join(data_dir, "household_power_consumption.txt")
    
    # Check if file already exists
    if os.path.exists(txt_path):
        logger.info(f"Data file already exists at {txt_path}")
        return txt_path
    
    logger.info(f"Downloading data from {url}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Extract zip file
        logger.info("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Remove zip file
        os.remove(zip_path)
        
        logger.info(f"Data downloaded and extracted to {txt_path}")
        return txt_path
        
    except Exception as e:
        logger.error(f"Error downloading data: {str(e)}")
        raise


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw power consumption data from CSV file.
    
    Args:
        file_path: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    logger.info(f"Loading raw data from {file_path}")
    
    try:
        # Read CSV with proper parsing
        df = pd.read_csv(
            file_path,
            sep=';',
            parse_dates={'datetime': ['Date', 'Time']},
            date_parser=lambda x: pd.to_datetime(x, format='%d/%m/%Y %H:%M:%S'),
            low_memory=False
        )
        
        logger.info(f"Loaded {len(df)} records from {file_path}")
        logger.info(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare data for ML.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    logger.info("Cleaning data...")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle missing values (marked as '?')
    logger.info("Handling missing values...")
    df_clean = df_clean.replace('?', np.nan)
    
    # Convert numeric columns to proper types
    numeric_cols = [
        'Global_active_power', 
        'Global_reactive_power', 
        'Voltage', 
        'Global_intensity',
        'Sub_metering_1',
        'Sub_metering_2',
        'Sub_metering_3'
    ]
    
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Log missing values
    missing_counts = df_clean.isnull().sum()
    logger.info(f"Missing values per column:\n{missing_counts[missing_counts > 0]}")
    
    # Feature engineering: total sub-metering
    df_clean['Total_sub_metering'] = (
        df_clean['Sub_metering_1'] + 
        df_clean['Sub_metering_2'] + 
        df_clean['Sub_metering_3']
    )
    
    # Remove rows with missing target variable
    initial_len = len(df_clean)
    df_clean = df_clean.dropna(subset=['Global_active_power'])
    final_len = len(df_clean)
    
    logger.info(f"Removed {initial_len - final_len} rows with missing target values")
    logger.info(f"Final dataset shape: {df_clean.shape}")
    
    return df_clean


def validate_data(df: pd.DataFrame) -> bool:
    """
    Validate data quality and structure.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if validation passes
    """
    logger.info("Validating data quality...")
    
    # Check required columns
    required_cols = [
        'datetime', 'Global_active_power', 'Global_reactive_power', 
        'Voltage', 'Global_intensity', 'Sub_metering_1', 
        'Sub_metering_2', 'Sub_metering_3'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False
    
    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        logger.error("datetime column is not datetime type")
        return False
    
    # Check for reasonable value ranges
    if df['Global_active_power'].max() > 20:  # Reasonable max for household
        logger.warning("Unusually high power consumption values detected")
    
    if df['Voltage'].min() < 200 or df['Voltage'].max() > 250:
        logger.warning("Voltage values outside expected range (200-250V)")
    
    # Check data completeness
    completeness = (1 - df.isnull().sum() / len(df)) * 100
    logger.info(f"Data completeness:\n{completeness}")
    
    logger.info("Data validation completed")
    return True


def save_processed_data(df: pd.DataFrame, file_path: str) -> None:
    """
    Save processed data to CSV file.
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
    """
    logger.info(f"Saving processed data to {file_path}")
    
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    logger.info(f"Processed data saved to {file_path}")


if __name__ == "__main__":
    # Example usage
    try:
        # Download data (if not already present)
        data_path = download_data()
        
        # Load raw data
        df = load_raw_data(data_path)
        
        # Clean data
        df_clean = clean_data(df)
        
        # Validate data
        if validate_data(df_clean):
            # Save processed data
            save_processed_data(df_clean, "data/processed/processed_energy_data.csv")
            logger.info("Data processing completed successfully!")
        else:
            logger.error("Data validation failed!")
            
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        raise
