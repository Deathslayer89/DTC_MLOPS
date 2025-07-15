#!/usr/bin/env python3
"""
Setup script for Smart Energy Prediction project.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(command, check=True):
    """Run a command and handle errors."""
    logger.info(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0 and check:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {result.stderr}")
        sys.exit(1)
    
    return result


def setup_environment():
    """Setup Python environment."""
    logger.info("Setting up Python environment...")
    
    # Create virtual environment
    if not os.path.exists("venv"):
        run_command("python -m venv venv")
        logger.info("Created virtual environment")
    
    # Activate virtual environment and install dependencies
    if os.name == 'nt':  # Windows
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_cmd = "source venv/bin/activate"
    
    run_command(f"{activate_cmd} && pip install --upgrade pip")
    run_command(f"{activate_cmd} && pip install -r requirements.txt")
    
    logger.info("Installed Python dependencies")


def setup_directories():
    """Setup project directories."""
    logger.info("Setting up project directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "reports",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def setup_git_hooks():
    """Setup Git hooks."""
    logger.info("Setting up Git hooks...")
    
    try:
        # Install pre-commit hooks
        if os.name == 'nt':  # Windows
            activate_cmd = "venv\\Scripts\\activate"
        else:  # Unix/Linux/MacOS
            activate_cmd = "source venv/bin/activate"
        
        run_command(f"{activate_cmd} && pre-commit install")
        logger.info("Installed pre-commit hooks")
    except Exception as e:
        logger.warning(f"Failed to install pre-commit hooks: {e}")


def check_dataset():
    """Check if dataset exists."""
    logger.info("Checking for dataset...")
    
    dataset_path = "data/raw/household_power_consumption.txt"
    if os.path.exists(dataset_path):
        logger.info("Dataset found!")
        return True
    else:
        logger.warning("Dataset not found!")
        logger.info("Please download the dataset from:")
        logger.info("https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip")
        logger.info("Extract household_power_consumption.txt to data/raw/")
        return False


def setup_mlflow():
    """Setup MLflow."""
    logger.info("Setting up MLflow...")
    
    # Create MLflow directory
    Path("mlruns").mkdir(exist_ok=True)
    
    logger.info("MLflow setup complete")


def display_next_steps():
    """Display next steps for the user."""
    logger.info("\n" + "="*50)
    logger.info("SETUP COMPLETE!")
    logger.info("="*50)
    
    logger.info("\nNext steps:")
    logger.info("1. Download the dataset:")
    logger.info("   wget https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip")
    logger.info("   unzip household_power_consumption.zip")
    logger.info("   mv household_power_consumption.txt data/raw/")
    
    logger.info("\n2. Process the data:")
    logger.info("   python scripts/process_data.py")
    
    logger.info("\n3. Train the model:")
    logger.info("   python src/models/train_model.py")
    
    logger.info("\n4. Start MLflow UI:")
    logger.info("   mlflow ui")
    
    logger.info("\n5. Start the API:")
    logger.info("   python src/api/app.py")
    
    logger.info("\n6. Run monitoring:")
    logger.info("   python src/monitoring/model_monitor.py")
    
    logger.info("\n7. Run tests:")
    logger.info("   pytest tests/ -v")
    
    logger.info("\nFor more information, see README.md")


def main():
    """Main setup function."""
    try:
        logger.info("Starting Smart Energy Prediction project setup...")
        
        # Setup directories
        setup_directories()
        
        # Setup Python environment
        setup_environment()
        
        # Setup Git hooks
        setup_git_hooks()
        
        # Setup MLflow
        setup_mlflow()
        
        # Check for dataset
        dataset_exists = check_dataset()
        
        # Display next steps
        display_next_steps()
        
        logger.info("Setup completed successfully!")
        
        if not dataset_exists:
            logger.warning("Don't forget to download the dataset!")
        
    except Exception as e:
        logger.error(f"Setup failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
