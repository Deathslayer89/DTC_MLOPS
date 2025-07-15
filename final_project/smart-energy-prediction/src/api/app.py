"""
FastAPI application for Smart Energy Prediction API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import mlflow.pyfunc
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
import uvicorn
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Smart Energy Prediction API",
    description="API for predicting household energy consumption",
    version="1.0.0"
)

# Global variables for model and scaler
model = None
scaler = None


class PredictionRequest(BaseModel):
    """Request model for energy prediction."""
    
    Global_reactive_power: float = Field(..., description="Global reactive power (kW)")
    Voltage: float = Field(..., description="Voltage (V)")
    Global_intensity: float = Field(..., description="Global intensity (A)")
    Sub_metering_1: float = Field(..., description="Kitchen sub-metering (Wh)")
    Sub_metering_2: float = Field(..., description="Laundry sub-metering (Wh)")
    Sub_metering_3: float = Field(..., description="Water heater & AC sub-metering (Wh)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    day_of_month: int = Field(..., ge=1, le=31, description="Day of month (1-31)")
    quarter: int = Field(..., ge=1, le=4, description="Quarter (1-4)")
    year: int = Field(..., description="Year")
    
    class Config:
        schema_extra = {
            "example": {
                "Global_reactive_power": 0.1,
                "Voltage": 240.0,
                "Global_intensity": 4.0,
                "Sub_metering_1": 0.0,
                "Sub_metering_2": 1.0,
                "Sub_metering_3": 17.0,
                "hour": 14,
                "day_of_week": 1,
                "month": 6,
                "day_of_month": 15,
                "quarter": 2,
                "year": 2024
            }
        }


class PredictionResponse(BaseModel):
    """Response model for energy prediction."""
    
    prediction: float = Field(..., description="Predicted energy consumption (kW)")
    confidence_interval: List[float] = Field(..., description="Confidence interval [lower, upper]")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    
    class Config:
        schema_extra = {
            "example": {
                "prediction": 2.5,
                "confidence_interval": [2.2, 2.8],
                "timestamp": "2024-01-01T14:30:00",
                "model_version": "1.0.0"
            }
        }


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: str = Field(..., description="Model version")
    timestamp: str = Field(..., description="Health check timestamp")


def load_model_and_scaler():
    """Load the trained model and scaler."""
    global model, scaler
    
    try:
        # Try to load MLflow model first
        model_uri = "models:/energy-consumption-model/Production"
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            logger.info(f"Loaded MLflow model from {model_uri}")
        except Exception as e:
            logger.warning(f"Could not load MLflow model: {str(e)}")
            
            # Fallback to joblib model
            model_path = "models/best_model.pkl"
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                logger.info(f"Loaded joblib model from {model_path}")
            else:
                logger.error("No model found!")
                raise FileNotFoundError("No model found")
        
        # Load scaler
        scaler_path = "models/scaler.pkl"
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning("No scaler found - predictions may be less accurate")
            
    except Exception as e:
        logger.error(f"Error loading model and scaler: {str(e)}")
        raise


def engineer_features(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for prediction.
    
    Args:
        input_data: Input DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    df_features = input_data.copy()
    
    # Create derived features
    df_features['Total_sub_metering'] = (
        df_features['Sub_metering_1'] + 
        df_features['Sub_metering_2'] + 
        df_features['Sub_metering_3']
    )
    
    # Season mapping
    season_mapping = {
        12: 0, 1: 0, 2: 0,  # Winter
        3: 1, 4: 1, 5: 1,   # Spring
        6: 2, 7: 2, 8: 2,   # Summer
        9: 3, 10: 3, 11: 3  # Fall
    }
    df_features['season'] = df_features['month'].map(season_mapping)
    
    # Time-based features
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    df_features['is_working_hours'] = (
        (df_features['hour'] >= 9) & 
        (df_features['hour'] <= 17) & 
        (df_features['day_of_week'] < 5)
    ).astype(int)
    
    # Power efficiency (simplified)
    df_features['power_efficiency'] = (
        df_features['Global_intensity'] / (df_features['Voltage'] + 1e-8)
    )
    
    # Sub-metering ratios
    total_submetering = df_features['Total_sub_metering']
    df_features['kitchen_ratio'] = (
        df_features['Sub_metering_1'] / (total_submetering + 1e-8)
    )
    df_features['laundry_ratio'] = (
        df_features['Sub_metering_2'] / (total_submetering + 1e-8)
    )
    df_features['hvac_ratio'] = (
        df_features['Sub_metering_3'] / (total_submetering + 1e-8)
    )
    
    # Peak usage indicator
    df_features['is_peak_hour'] = (
        (df_features['hour'].isin([8, 9, 18, 19, 20])) & 
        (df_features['day_of_week'] < 5)
    ).astype(int)
    
    # Time of day categorical (one-hot encoded)
    def categorize_time(hour):
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    time_of_day = df_features['hour'].apply(categorize_time)
    
    # One-hot encode time_of_day
    df_features['time_of_day_afternoon'] = (time_of_day == 'afternoon').astype(int)
    df_features['time_of_day_evening'] = (time_of_day == 'evening').astype(int)
    df_features['time_of_day_morning'] = (time_of_day == 'morning').astype(int)
    df_features['time_of_day_night'] = (time_of_day == 'night').astype(int)
    
    # Add missing features with default values (for lag and rolling features)
    # These would normally be computed from historical data
    missing_features = [
        'Global_active_power_lag_1', 'Global_active_power_lag_24', 'Global_active_power_lag_168',
        'Voltage_lag_1', 'Voltage_lag_24', 'Global_intensity_lag_1', 'Global_intensity_lag_24',
        'Global_active_power_rolling_mean_24', 'Global_active_power_rolling_std_24',
        'Global_active_power_rolling_min_24', 'Global_active_power_rolling_max_24',
        'Global_active_power_rolling_mean_168', 'Global_active_power_rolling_std_168',
        'Global_active_power_rolling_min_168', 'Global_active_power_rolling_max_168',
        'Global_active_power_rolling_mean_720', 'Global_active_power_rolling_std_720',
        'Global_active_power_rolling_min_720', 'Global_active_power_rolling_max_720',
        'voltage_stability'
    ]
    
    for feature in missing_features:
        df_features[feature] = 0.0  # Default value
    
    return df_features


@app.on_event("startup")
async def startup_event():
    """Load model and scaler on startup."""
    load_model_and_scaler()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_energy_consumption(request: PredictionRequest):
    """
    Predict energy consumption.
    
    Args:
        request: Prediction request
        
    Returns:
        Prediction response
    """
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Engineer features
        input_features = engineer_features(input_data)
        
        # Scale features if scaler is available
        if scaler is not None:
            # Derive expected feature order directly from the trained scaler when available.
            # This guarantees that inference uses the **exact** column order from training,
            # preventing the mismatch error we observed.
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                expected_features = list(scaler.feature_names_in_)
            else:
                # Fallback to the previously hard-coded list (kept for backward compatibility)
                expected_features = [
                    'Global_reactive_power', 'Voltage', 'Global_intensity', 
                    'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 
                    'Total_sub_metering', 'hour', 'day_of_week', 'month', 
                    'day_of_month', 'quarter', 'year', 'season', 'is_weekend', 
                    'is_working_hours', 'power_efficiency', 'kitchen_ratio', 
                    'laundry_ratio', 'hvac_ratio', 'is_peak_hour',
                    'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_morning',
                    'time_of_day_night', 'Global_active_power_lag_1', 'Global_active_power_lag_24',
                    'Global_active_power_lag_168', 'Voltage_lag_1', 'Voltage_lag_24',
                    'Global_intensity_lag_1', 'Global_intensity_lag_24',
                    'Global_active_power_rolling_mean_24', 'Global_active_power_rolling_std_24',
                    'Global_active_power_rolling_min_24', 'Global_active_power_rolling_max_24',
                    'Global_active_power_rolling_mean_168', 'Global_active_power_rolling_std_168',
                    'Global_active_power_rolling_min_168', 'Global_active_power_rolling_max_168',
                    'Global_active_power_rolling_mean_720', 'Global_active_power_rolling_std_720',
                    'Global_active_power_rolling_min_720', 'Global_active_power_rolling_max_720',
                    'voltage_stability'
                ]
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in input_features.columns:
                    input_features[feature] = 0.0
            
            # Reorder columns to match training
            input_features = input_features[expected_features]
            
            # Scale features
            input_features_scaled = scaler.transform(input_features)
            input_features = pd.DataFrame(input_features_scaled, columns=expected_features)
        
        # Make prediction
        prediction = model.predict(input_features)[0]
        
        # Calculate confidence interval (simplified approach)
        # In practice, you might use prediction intervals from the model
        confidence_interval = [
            max(0, prediction * 0.9),  # Ensure non-negative
            prediction * 1.1
        ]
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            timestamp=datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Smart Energy Prediction API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/model/info")
async def model_info():
    """Get model information."""
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_version": "1.0.0",
        "model_type": type(model).__name__ if model else None
    }


if __name__ == "__main__":
    # Run the application
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False
    )
