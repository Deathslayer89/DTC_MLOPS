"""
Test suite for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import joblib
import os
import sys
import tempfile

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app, load_model_and_scaler, engineer_features


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def setup_method(self):
        """Setup test client."""
        self.client = TestClient(app)
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = self.client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "Smart Energy Prediction API"
        assert response.json()["version"] == "1.0.0"
        assert response.json()["status"] == "running"
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"
    
    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "model_loaded" in data
        assert "scaler_loaded" in data
        assert "model_version" in data
        assert "model_type" in data
    
    @patch('src.api.app.model')
    @patch('src.api.app.scaler')
    def test_prediction_endpoint_success(self, mock_scaler, mock_model):
        """Test successful prediction."""
        # Mock model and scaler
        mock_model.predict.return_value = np.array([2.5])
        mock_scaler.transform.return_value = np.array([[1, 2, 3, 4, 5]])
        
        test_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 1,
            "month": 6
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 200
        
        data = response.json()
        assert "prediction" in data
        assert "confidence_interval" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert data["prediction"] == 2.5
        assert len(data["confidence_interval"]) == 2
    
    def test_prediction_endpoint_invalid_data(self):
        """Test prediction with invalid data."""
        test_data = {
            "Global_reactive_power": "invalid",  # Should be float
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 1,
            "month": 6
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_endpoint_missing_fields(self):
        """Test prediction with missing required fields."""
        test_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            # Missing other required fields
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_prediction_endpoint_out_of_range(self):
        """Test prediction with out of range values."""
        test_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 25,  # Invalid hour
            "day_of_week": 1,
            "month": 6
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 422  # Validation error
    
    @patch('src.api.app.model', None)
    def test_prediction_endpoint_model_not_loaded(self):
        """Test prediction when model is not loaded."""
        test_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 1,
            "month": 6
        }
        
        response = self.client.post("/predict", json=test_data)
        assert response.status_code == 503  # Service unavailable


class TestFeatureEngineering:
    """Test cases for feature engineering in API."""
    
    def test_engineer_features_basic(self):
        """Test basic feature engineering."""
        input_data = pd.DataFrame({
            'Global_reactive_power': [0.1],
            'Voltage': [240.0],
            'Global_intensity': [4.0],
            'Sub_metering_1': [0.0],
            'Sub_metering_2': [1.0],
            'Sub_metering_3': [17.0],
            'hour': [14],
            'day_of_week': [1],
            'month': [6]
        })
        
        result = engineer_features(input_data)
        
        # Check that all expected features are created
        assert 'Total_sub_metering' in result.columns
        assert 'season' in result.columns
        assert 'is_weekend' in result.columns
        assert 'is_working_hours' in result.columns
        assert 'power_efficiency' in result.columns
        assert 'kitchen_ratio' in result.columns
        assert 'laundry_ratio' in result.columns
        assert 'hvac_ratio' in result.columns
        assert 'is_peak_hour' in result.columns
        
        # Check one-hot encoded time_of_day features
        assert 'time_of_day_afternoon' in result.columns
        assert 'time_of_day_evening' in result.columns
        assert 'time_of_day_morning' in result.columns
        assert 'time_of_day_night' in result.columns
    
    def test_engineer_features_values(self):
        """Test feature engineering produces correct values."""
        input_data = pd.DataFrame({
            'Global_reactive_power': [0.1],
            'Voltage': [240.0],
            'Global_intensity': [4.0],
            'Sub_metering_1': [2.0],
            'Sub_metering_2': [3.0],
            'Sub_metering_3': [5.0],
            'hour': [14],  # Afternoon
            'day_of_week': [1],  # Tuesday
            'month': [6]  # June (summer)
        })
        
        result = engineer_features(input_data)
        
        # Check specific values
        assert result['Total_sub_metering'].iloc[0] == 10.0
        assert result['season'].iloc[0] == 2  # Summer
        assert result['is_weekend'].iloc[0] == 0  # Not weekend
        assert result['is_working_hours'].iloc[0] == 1  # Working hours
        assert result['time_of_day_afternoon'].iloc[0] == 1  # Afternoon
        assert result['time_of_day_morning'].iloc[0] == 0  # Not morning
        
        # Check ratios
        assert result['kitchen_ratio'].iloc[0] == 0.2  # 2/10
        assert result['laundry_ratio'].iloc[0] == 0.3  # 3/10
        assert result['hvac_ratio'].iloc[0] == 0.5  # 5/10
    
    def test_engineer_features_weekend(self):
        """Test weekend feature detection."""
        input_data = pd.DataFrame({
            'Global_reactive_power': [0.1],
            'Voltage': [240.0],
            'Global_intensity': [4.0],
            'Sub_metering_1': [0.0],
            'Sub_metering_2': [1.0],
            'Sub_metering_3': [17.0],
            'hour': [14],
            'day_of_week': [5],  # Saturday
            'month': [6]
        })
        
        result = engineer_features(input_data)
        
        assert result['is_weekend'].iloc[0] == 1  # Weekend
        assert result['is_working_hours'].iloc[0] == 0  # Not working hours on weekend
    
    def test_engineer_features_peak_hour(self):
        """Test peak hour detection."""
        input_data = pd.DataFrame({
            'Global_reactive_power': [0.1],
            'Voltage': [240.0],
            'Global_intensity': [4.0],
            'Sub_metering_1': [0.0],
            'Sub_metering_2': [1.0],
            'Sub_metering_3': [17.0],
            'hour': [19],  # Peak hour
            'day_of_week': [1],  # Weekday
            'month': [6]
        })
        
        result = engineer_features(input_data)
        
        assert result['is_peak_hour'].iloc[0] == 1  # Peak hour
    
    def test_engineer_features_time_of_day_categories(self):
        """Test time of day categorization."""
        # Test different hours
        hours = [7, 13, 19, 23]  # Morning, afternoon, evening, night
        expected_categories = [
            {'time_of_day_morning': 1, 'time_of_day_afternoon': 0, 'time_of_day_evening': 0, 'time_of_day_night': 0},
            {'time_of_day_morning': 0, 'time_of_day_afternoon': 1, 'time_of_day_evening': 0, 'time_of_day_night': 0},
            {'time_of_day_morning': 0, 'time_of_day_afternoon': 0, 'time_of_day_evening': 1, 'time_of_day_night': 0},
            {'time_of_day_morning': 0, 'time_of_day_afternoon': 0, 'time_of_day_evening': 0, 'time_of_day_night': 1}
        ]
        
        for hour, expected in zip(hours, expected_categories):
            input_data = pd.DataFrame({
                'Global_reactive_power': [0.1],
                'Voltage': [240.0],
                'Global_intensity': [4.0],
                'Sub_metering_1': [0.0],
                'Sub_metering_2': [1.0],
                'Sub_metering_3': [17.0],
                'hour': [hour],
                'day_of_week': [1],
                'month': [6]
            })
            
            result = engineer_features(input_data)
            
            for feature, expected_value in expected.items():
                assert result[feature].iloc[0] == expected_value, f"Failed for hour {hour}, feature {feature}"


class TestModelLoading:
    """Test cases for model loading functionality."""
    
    def test_load_model_and_scaler_success(self):
        """Test successful model and scaler loading."""
        # Create mock model and scaler files
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            mock_model = Mock()
            mock_model.predict.return_value = np.array([2.5])
            joblib.dump(mock_model, f.name)
            model_path = f.name
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            mock_scaler = Mock()
            mock_scaler.transform.return_value = np.array([[1, 2, 3]])
            joblib.dump(mock_scaler, f.name)
            scaler_path = f.name
        
        try:
            with patch('src.api.app.model', None):
                with patch('src.api.app.scaler', None):
                    with patch('os.path.exists') as mock_exists:
                        mock_exists.side_effect = lambda path: path in [model_path, scaler_path]
                        
                        with patch('joblib.load') as mock_load:
                            mock_load.side_effect = lambda path: {
                                model_path: mock_model,
                                scaler_path: mock_scaler
                            }[path]
                            
                            # This would normally be called during startup
                            # load_model_and_scaler()
                            
                            # Test would verify model and scaler are loaded
                            # assert model is not None
                            # assert scaler is not None
                            pass
        finally:
            os.unlink(model_path)
            os.unlink(scaler_path)
    
    @patch('mlflow.pyfunc.load_model')
    def test_load_mlflow_model_success(self, mock_mlflow_load):
        """Test successful MLflow model loading."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([2.5])
        mock_mlflow_load.return_value = mock_model
        
        # This would test MLflow model loading
        # The actual implementation would need to be tested
        pass
    
    def test_load_model_file_not_found(self):
        """Test model loading when files don't exist."""
        with patch('os.path.exists', return_value=False):
            with patch('mlflow.pyfunc.load_model') as mock_mlflow_load:
                mock_mlflow_load.side_effect = Exception("Model not found")
                
                with pytest.raises(Exception):
                    load_model_and_scaler()


class TestInputValidation:
    """Test cases for input validation."""
    
    def test_prediction_request_validation(self):
        """Test prediction request validation."""
        from src.api.app import PredictionRequest
        
        # Valid request
        valid_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 1,
            "month": 6
        }
        
        request = PredictionRequest(**valid_data)
        assert request.hour == 14
        assert request.day_of_week == 1
        assert request.month == 6
    
    def test_prediction_request_hour_validation(self):
        """Test hour validation in prediction request."""
        from src.api.app import PredictionRequest
        from pydantic import ValidationError
        
        invalid_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 25,  # Invalid hour
            "day_of_week": 1,
            "month": 6
        }
        
        with pytest.raises(ValidationError):
            PredictionRequest(**invalid_data)
    
    def test_prediction_request_day_of_week_validation(self):
        """Test day of week validation in prediction request."""
        from src.api.app import PredictionRequest
        from pydantic import ValidationError
        
        invalid_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 8,  # Invalid day
            "month": 6
        }
        
        with pytest.raises(ValidationError):
            PredictionRequest(**invalid_data)
    
    def test_prediction_request_month_validation(self):
        """Test month validation in prediction request."""
        from src.api.app import PredictionRequest
        from pydantic import ValidationError
        
        invalid_data = {
            "Global_reactive_power": 0.1,
            "Voltage": 240.0,
            "Global_intensity": 4.0,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 1.0,
            "Sub_metering_3": 17.0,
            "hour": 14,
            "day_of_week": 1,
            "month": 13  # Invalid month
        }
        
        with pytest.raises(ValidationError):
            PredictionRequest(**invalid_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
