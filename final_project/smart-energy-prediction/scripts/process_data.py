#!/usr/bin/env python3
"""
Data processing script for Smart Energy Prediction project.
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.data_loader import load_raw_data, clean_data, validate_data, save_processed_data
from src.features.feature_engineering import engineer_features

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main data processing function."""
    try:
        logger.info("Starting data processing pipeline...")
        
        # Check if raw data exists
        raw_data_path = "data/raw/household_power_consumption.txt"
        if not os.path.exists(raw_data_path):
            logger.error(f"Raw data file not found at {raw_data_path}")
            logger.error("Please download the dataset from:")
            logger.error("https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip")
            logger.error("Extract household_power_consumption.txt to data/raw/")
            sys.exit(1)
        
        # Load raw data
        logger.info("Loading raw data...")
        df = load_raw_data(raw_data_path)
        logger.info(f"Loaded {len(df)} records")
        
        # Clean data
        logger.info("Cleaning data...")
        df_clean = clean_data(df)
        logger.info(f"Cleaned data shape: {df_clean.shape}")
        
        # Validate data
        logger.info("Validating data...")
        if not validate_data(df_clean):
            logger.error("Data validation failed!")
            sys.exit(1)
        
        # Engineer features
        logger.info("Engineering features...")
        df_features = engineer_features(df_clean)
        logger.info(f"Feature engineered data shape: {df_features.shape}")
        
        # Save processed data
        processed_path = "data/processed/processed_energy_data.csv"
        logger.info(f"Saving processed data to {processed_path}")
        save_processed_data(df_features, processed_path)
        
        # Create reference data for monitoring
        reference_path = "data/processed/reference_data.csv"
        logger.info(f"Creating reference data at {reference_path}")
        reference_data = df_features.sample(n=min(10000, len(df_features)), random_state=42)
        save_processed_data(reference_data, reference_path)
        
        logger.info("Data processing completed successfully!")
        logger.info(f"Original data: {len(df)} records")
        logger.info(f"Processed data: {len(df_features)} records")
        logger.info(f"Features: {df_features.shape[1]} columns")
        
    except Exception as e:
        logger.error(f"Data processing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
