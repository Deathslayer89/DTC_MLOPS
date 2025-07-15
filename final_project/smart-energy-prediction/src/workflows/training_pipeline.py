"""
Prefect workflow orchestration for Smart Energy Prediction project.
"""

from prefect import flow, task, get_run_logger
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import os
import logging

# Import project modules
from src.data.data_loader import load_raw_data, clean_data, validate_data, save_processed_data
from src.features.feature_engineering import engineer_features, select_features, split_data, scale_features
from src.models.train_model import ModelTrainer
from src.monitoring.model_monitor import ModelMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)


@task(retry_delay_seconds=60, retries=3)
def extract_data(file_path: str = "data/raw/household_power_consumption.txt") -> pd.DataFrame:
    """
    Extract data from source.
    
    Args:
        file_path: Path to raw data file
        
    Returns:
        Raw DataFrame
    """
    logger = get_run_logger()
    logger.info(f"Extracting data from {file_path}")
    
    try:
        df = load_raw_data(file_path)
        logger.info(f"Successfully extracted {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Data extraction failed: {str(e)}")
        raise


@task
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform and engineer features.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Transformed DataFrame
    """
    logger = get_run_logger()
    logger.info("Transforming data...")
    
    try:
        # Clean data
        df_clean = clean_data(df)
        logger.info("Data cleaning completed")
        
        # Validate data
        if not validate_data(df_clean):
            raise ValueError("Data validation failed")
        logger.info("Data validation passed")
        
        # Engineer features
        df_features = engineer_features(df_clean)
        logger.info("Feature engineering completed")
        
        return df_features
        
    except Exception as e:
        logger.error(f"Data transformation failed: {str(e)}")
        raise


@task
def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                   pd.Series, pd.Series, pd.Series, Any]:
    """
    Prepare data for training.
    
    Args:
        df: Transformed DataFrame
        
    Returns:
        Tuple of training data and scaler
    """
    logger = get_run_logger()
    logger.info("Preparing training data...")
    
    try:
        # Select features
        X, y = select_features(df)
        logger.info(f"Selected {X.shape[1]} features for training")
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        logger.info("Data split completed")
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
        logger.info("Feature scaling completed")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler
        
    except Exception as e:
        logger.error(f"Training data preparation failed: {str(e)}")
        raise


@task
def train_model_task(X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Any, Dict[str, float]]:
    """
    Train model with MLflow tracking.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Tuple of (best_model, metrics)
    """
    logger = get_run_logger()
    logger.info("Training model...")
    
    try:
        # Initialize trainer
        trainer = ModelTrainer()
        
        # Train best model
        best_model, best_metrics = trainer.train_best_model(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        
        logger.info(f"Model training completed - Best MAE: {best_metrics['mae']:.4f}")
        
        return best_model, best_metrics
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise


@task
def validate_model_performance(model: Any, X_test: pd.DataFrame, y_test: pd.Series,
                             performance_threshold: float = 0.5) -> Dict[str, float]:
    """
    Validate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        performance_threshold: MAE threshold for acceptable performance
        
    Returns:
        Model metrics
    """
    logger = get_run_logger()
    logger.info("Validating model performance...")
    
    try:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        predictions = model.predict(X_test)
        
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'mse': mean_squared_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        
        # Check if performance is acceptable
        if metrics['mae'] > performance_threshold:
            logger.warning(f"Model performance degraded: MAE = {metrics['mae']:.4f}")
            raise ValueError(f"Model performance degraded: MAE = {metrics['mae']:.4f}")
        
        logger.info(f"Model validation passed - MAE: {metrics['mae']:.4f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise


@task
def save_model_artifacts(model: Any, scaler: Any, metrics: Dict[str, float]) -> str:
    """
    Save model artifacts.
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        metrics: Model metrics
        
    Returns:
        Path to saved artifacts
    """
    logger = get_run_logger()
    logger.info("Saving model artifacts...")
    
    try:
        # Save model and scaler
        trainer = ModelTrainer()
        trainer.save_model(model, scaler)
        
        # Save metrics
        import json
        metrics_path = "models/metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info("Model artifacts saved successfully")
        
        return "models/"
        
    except Exception as e:
        logger.error(f"Saving model artifacts failed: {str(e)}")
        raise


@task
def save_training_data(df: pd.DataFrame, save_path: str = "data/processed/processed_energy_data.csv") -> str:
    """
    Save processed training data.
    
    Args:
        df: Processed DataFrame
        save_path: Path to save data
        
    Returns:
        Path to saved data
    """
    logger = get_run_logger()
    logger.info(f"Saving processed data to {save_path}")
    
    try:
        save_processed_data(df, save_path)
        logger.info("Processed data saved successfully")
        return save_path
        
    except Exception as e:
        logger.error(f"Saving processed data failed: {str(e)}")
        raise


@task
def run_monitoring_task() -> Dict[str, Any]:
    """
    Run model monitoring.
    
    Returns:
        Monitoring results
    """
    logger = get_run_logger()
    logger.info("Running model monitoring...")
    
    try:
        # Initialize monitor
        monitor = ModelMonitor()
        
        # Run monitoring pipeline
        results = monitor.run_monitoring_pipeline()
        
        logger.info("Model monitoring completed successfully")
        
        return results
        
    except Exception as e:
        logger.error(f"Model monitoring failed: {str(e)}")
        raise


@flow(name="energy-prediction-training", task_runner=SequentialTaskRunner())
def training_pipeline():
    """
    Complete training pipeline.
    
    Returns:
        Training results
    """
    logger = get_run_logger()
    logger.info("Starting energy prediction training pipeline...")
    
    try:
        # Extract data
        raw_data = extract_data()
        
        # Transform data
        processed_data = transform_data(raw_data)
        
        # Save processed data
        data_path = save_training_data(processed_data)
        
        # Prepare training data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_training_data(processed_data)
        
        # Train model
        model, metrics = train_model_task(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Validate performance
        final_metrics = validate_model_performance(model, X_test, y_test)
        
        # Save artifacts
        model_path = save_model_artifacts(model, scaler, final_metrics)
        
        logger.info("Training pipeline completed successfully!")
        
        return {
            'status': 'success',
            'metrics': final_metrics,
            'model_path': model_path,
            'data_path': data_path
        }
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


@flow(name="energy-prediction-monitoring", task_runner=SequentialTaskRunner())
def monitoring_pipeline():
    """
    Complete monitoring pipeline.
    
    Returns:
        Monitoring results
    """
    logger = get_run_logger()
    logger.info("Starting energy prediction monitoring pipeline...")
    
    try:
        # Run monitoring
        results = run_monitoring_task()
        
        logger.info("Monitoring pipeline completed successfully!")
        
        return {
            'status': 'success',
            'monitoring_results': results
        }
        
    except Exception as e:
        logger.error(f"Monitoring pipeline failed: {str(e)}")
        raise


@flow(name="energy-prediction-full-pipeline", task_runner=SequentialTaskRunner())
def full_pipeline():
    """
    Complete ML pipeline with training and monitoring.
    
    Returns:
        Pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting full energy prediction pipeline...")
    
    try:
        # Run training pipeline
        training_results = training_pipeline()
        
        # Run monitoring pipeline
        monitoring_results = monitoring_pipeline()
        
        logger.info("Full pipeline completed successfully!")
        
        return {
            'status': 'success',
            'training_results': training_results,
            'monitoring_results': monitoring_results
        }
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {str(e)}")
        raise


# Deployment configurations
def create_deployments():
    """Create Prefect deployments."""
    
    # Daily training deployment
    training_deployment = Deployment.build_from_flow(
        flow=training_pipeline,
        name="daily-energy-model-training",
        schedule=CronSchedule(cron="0 2 * * *"),  # Daily at 2 AM
        tags=["energy", "training", "production"]
    )
    
    # Hourly monitoring deployment
    monitoring_deployment = Deployment.build_from_flow(
        flow=monitoring_pipeline,
        name="hourly-energy-model-monitoring",
        schedule=CronSchedule(cron="0 * * * *"),  # Every hour
        tags=["energy", "monitoring", "production"]
    )
    
    # Weekly full pipeline deployment
    full_pipeline_deployment = Deployment.build_from_flow(
        flow=full_pipeline,
        name="weekly-energy-full-pipeline",
        schedule=CronSchedule(cron="0 0 * * 0"),  # Weekly on Sunday
        tags=["energy", "full-pipeline", "production"]
    )
    
    return [training_deployment, monitoring_deployment, full_pipeline_deployment]


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "training":
            training_pipeline()
        elif sys.argv[1] == "monitoring":
            monitoring_pipeline()
        elif sys.argv[1] == "full":
            full_pipeline()
        elif sys.argv[1] == "deploy":
            deployments = create_deployments()
            for deployment in deployments:
                deployment.apply()
            print("Deployments created successfully!")
        else:
            print("Usage: python training_pipeline.py [training|monitoring|full|deploy]")
    else:
        # Default: run training pipeline
        training_pipeline()
