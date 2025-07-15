"""
Model monitoring utilities with Evidently for Smart Energy Prediction project.
"""

import pandas as pd
import numpy as np
from evidently import Report
# from evidently.metric_preset import DataDriftPreset, RegressionPreset, DataQualityPreset
from evidently.metrics import (
    DataDriftMetric, 
    RegressionPerformanceMetric,
    DataQualityMetric
)
from evidently.test_suite import TestSuite
# from evidently.test_preset import DataDriftTestPreset, RegressionTestPreset, DataQualityTestPreset
from evidently.tests import (
    TestNumberOfColumnsWithMissingValues,
    TestNumberOfRowsWithMissingValues,
    TestNumberOfConstantColumns,
    TestNumberOfDuplicatedRows,
    TestDataDrift,
    TestMeanValueOfFeature
)
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelMonitor:
    """Model monitoring class using Evidently."""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Initialize ModelMonitor.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.report_path = self.config.get('monitoring', {}).get('report_path', 'reports/')
        
        # Ensure report directory exists
        os.makedirs(self.report_path, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {}
            
    def load_reference_data(self, file_path: str = "data/processed/reference_data.csv") -> pd.DataFrame:
        """
        Load reference dataset for drift detection.
        
        Args:
            file_path: Path to reference data
            
        Returns:
            Reference DataFrame
        """
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Loaded reference data from {file_path}")
                return df
            else:
                # Create reference data from processed data
                logger.info("Creating reference data from processed dataset...")
                processed_data = pd.read_csv("data/processed/processed_energy_data.csv")
                
                # Sample reference data
                reference_size = self.config.get('monitoring', {}).get('reference_data_size', 10000)
                reference_data = processed_data.sample(n=min(reference_size, len(processed_data)), 
                                                     random_state=42)
                
                # Save reference data
                reference_data.to_csv(file_path, index=False)
                logger.info(f"Created and saved reference data to {file_path}")
                
                return reference_data
                
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise
            
    def load_current_data(self, file_path: str = "data/processed/current_data.csv") -> pd.DataFrame:
        """
        Load current production data.
        
        Args:
            file_path: Path to current data
            
        Returns:
            Current DataFrame
        """
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                logger.info(f"Loaded current data from {file_path}")
                return df
            else:
                # For demo purposes, create current data by sampling from processed data
                logger.info("Creating current data from processed dataset...")
                processed_data = pd.read_csv("data/processed/processed_energy_data.csv")
                
                # Sample current data (different from reference)
                current_data = processed_data.sample(n=min(5000, len(processed_data)), 
                                                   random_state=123)
                
                # Add some simulated drift
                current_data = self._simulate_drift(current_data)
                
                # Save current data
                current_data.to_csv(file_path, index=False)
                logger.info(f"Created and saved current data to {file_path}")
                
                return current_data
                
        except Exception as e:
            logger.error(f"Error loading current data: {str(e)}")
            raise
            
    def _simulate_drift(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate data drift for demonstration purposes.
        
        Args:
            df: DataFrame to add drift to
            
        Returns:
            DataFrame with simulated drift
        """
        df_drift = df.copy()
        
        # Add drift to voltage (seasonal variation)
        df_drift['Voltage'] = df_drift['Voltage'] + np.random.normal(0, 2, len(df_drift))
        
        # Add drift to sub-metering patterns
        df_drift['Sub_metering_1'] = df_drift['Sub_metering_1'] * 1.1  # Kitchen usage increase
        
        # Add some outliers
        outlier_indices = np.random.choice(len(df_drift), size=int(len(df_drift) * 0.01), replace=False)
        df_drift.loc[outlier_indices, 'Global_active_power'] *= 3
        
        return df_drift
        
    def create_column_mapping(self) -> dict:
        """
        Create Evidently column mapping.
        
        Returns:
            Column mapping dict
        """
        return {
            'target': 'Global_active_power',
            'prediction': None,  # Will be added if predictions are available
            'numerical_features': [
                'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3',
                'Total_sub_metering', 'hour', 'day_of_week', 'month'
            ],
            'categorical_features': ['season', 'is_weekend', 'is_working_hours'],
            'datetime_features': ['datetime'] if 'datetime' in ['datetime'] else None
        }
        
    def generate_data_quality_report(self, reference_data: pd.DataFrame, 
                                   current_data: pd.DataFrame) -> Report:
        """
        Generate data quality report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Evidently Report
        """
        logger.info("Generating data quality report...")
        
        report = Report(metrics=[
            DataQualityMetric()
        ])
        
        report.run(reference_data=reference_data, 
                  current_data=current_data)
        
        return report
        
    def generate_data_drift_report(self, reference_data: pd.DataFrame, 
                                 current_data: pd.DataFrame) -> Report:
        """
        Generate data drift report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Evidently Report
        """
        logger.info("Generating data drift report...")
        
        report = Report(metrics=[
            DataDriftMetric()
        ])
        
        report.run(reference_data=reference_data, 
                  current_data=current_data)
        
        return report
        
    def generate_model_performance_report(self, reference_data: pd.DataFrame, 
                                        current_data: pd.DataFrame) -> Report:
        """
        Generate model performance report.
        
        Args:
            reference_data: Reference dataset with predictions
            current_data: Current dataset with predictions
            
        Returns:
            Evidently Report
        """
        logger.info("Generating model performance report...")
        
        report = Report(metrics=[
            RegressionPerformanceMetric()
        ])
        
        report.run(reference_data=reference_data, 
                  current_data=current_data)
        
        return report
        
    def run_test_suite(self, reference_data: pd.DataFrame, 
                      current_data: pd.DataFrame) -> TestSuite:
        """
        Run test suite for automated monitoring.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Evidently TestSuite
        """
        logger.info("Running test suite...")
        
        tests = TestSuite(tests=[
            TestNumberOfColumnsWithMissingValues(),
            TestNumberOfRowsWithMissingValues(),
            TestNumberOfConstantColumns(),
            TestNumberOfDuplicatedRows(),
            TestDataDrift(),
            TestMeanValueOfFeature(column_name='Global_active_power')
        ])
        
        tests.run(reference_data=reference_data, 
                 current_data=current_data)
        
        return tests
        
    def check_alerts(self, drift_report: Report, test_suite: TestSuite) -> List[Dict[str, Any]]:
        """
        Check for alerts and create alert messages.
        
        Args:
            drift_report: Data drift report
            test_suite: Test suite results
            
        Returns:
            List of alert messages
        """
        alerts = []
        
        try:
            # Check data drift
            drift_result = drift_report.as_dict()
            if 'metrics' in drift_result and len(drift_result['metrics']) > 0:
                drift_metric = drift_result['metrics'][0]
                if 'result' in drift_metric and 'dataset_drift' in drift_metric['result']:
                    drift_detected = drift_metric['result']['dataset_drift']
                    
                    if drift_detected:
                        alerts.append({
                            'type': 'data_drift',
                            'severity': 'warning',
                            'message': 'Data drift detected in energy consumption model!',
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Check test suite results
            test_results = test_suite.as_dict()
            if 'tests' in test_results:
                for test in test_results['tests']:
                    if test.get('status') == 'FAIL':
                        alerts.append({
                            'type': 'test_failure',
                            'severity': 'error',
                            'message': f"Test failed: {test.get('name', 'Unknown test')}",
                            'timestamp': datetime.now().isoformat()
                        })
            
            # Check for missing values
            if 'tests' in test_results:
                for test in test_results['tests']:
                    if 'missing_values' in test.get('name', '').lower() and test.get('status') == 'FAIL':
                        alerts.append({
                            'type': 'data_quality',
                            'severity': 'warning',
                            'message': 'High number of missing values detected',
                            'timestamp': datetime.now().isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            alerts.append({
                'type': 'system_error',
                'severity': 'error',
                'message': f'Error in monitoring system: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
        
    def send_alert(self, alert: Dict[str, Any]) -> None:
        """
        Send alert notification.
        
        Args:
            alert: Alert message dictionary
        """
        logger.warning(f"ALERT: {alert}")
        
        # For demo purposes, just log the alert
        # In production, you would send to Slack, email, etc.
        alert_message = f"[{alert['severity'].upper()}] {alert['message']}"
        
        if alert['severity'] == 'error':
            logger.error(alert_message)
        else:
            logger.warning(alert_message)
            
        # Here you could add:
        # - Email notifications
        # - Slack notifications
        # - PagerDuty alerts
        # - etc.
        
    def generate_monitoring_report(self, reference_data: pd.DataFrame, 
                                 current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive monitoring report.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            
        Returns:
            Dictionary with all monitoring results
        """
        logger.info("Generating comprehensive monitoring report...")
        
        try:
            # Generate reports
            quality_report = self.generate_data_quality_report(reference_data, current_data)
            drift_report = self.generate_data_drift_report(reference_data, current_data)
            test_suite = self.run_test_suite(reference_data, current_data)
            
            # Check for alerts
            alerts = self.check_alerts(drift_report, test_suite)
            
            # Send alerts
            for alert in alerts:
                self.send_alert(alert)
            
            # Save reports
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            quality_report.save_html(f"{self.report_path}/data_quality_report_{timestamp}.html")
            drift_report.save_html(f"{self.report_path}/data_drift_report_{timestamp}.html")
            test_suite.save_html(f"{self.report_path}/test_suite_report_{timestamp}.html")
            
            logger.info(f"Monitoring reports saved to {self.report_path}")
            
            return {
                'timestamp': timestamp,
                'alerts': alerts,
                'reports_saved': True,
                'report_path': self.report_path
            }
            
        except Exception as e:
            logger.error(f"Error generating monitoring report: {str(e)}")
            raise
            
    def run_monitoring_pipeline(self) -> Dict[str, Any]:
        """
        Run complete monitoring pipeline.
        
        Returns:
            Monitoring results
        """
        logger.info("Starting monitoring pipeline...")
        
        try:
            # Load data
            reference_data = self.load_reference_data()
            current_data = self.load_current_data()
            
            # Generate monitoring report
            results = self.generate_monitoring_report(reference_data, current_data)
            
            logger.info("Monitoring pipeline completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Monitoring pipeline failed: {str(e)}")
            raise


def main():
    """Main monitoring function."""
    try:
        # Initialize monitor
        monitor = ModelMonitor()
        
        # Run monitoring pipeline
        results = monitor.run_monitoring_pipeline()
        
        logger.info(f"Monitoring completed: {results}")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
