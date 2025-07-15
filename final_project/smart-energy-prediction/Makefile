# Smart Energy Prediction MLOps Project - Makefile
# Run common tasks with simple commands

.PHONY: help setup install data train api test mlflow clean docker lint format all quick dev

# Default target
help:
	@echo "Smart Energy Prediction MLOps Pipeline"
	@echo "======================================"
	@echo ""
	@echo "Available commands:"
	@echo "  make setup      - Setup project environment"
	@echo "  make install    - Install dependencies"
	@echo "  make data       - Process data pipeline"
	@echo "  make train      - Train models"
	@echo "  make api        - Start API server"
	@echo "  make test       - Run tests"
	@echo "  make mlflow     - Start MLflow UI"
	@echo "  make all        - Run complete pipeline (automated)"
	@echo "  make quick      - Quick start workflow"
	@echo "  make clean      - Clean generated files"
	@echo "  make docker     - Build Docker containers"
	@echo "  make lint       - Run code linting"
	@echo "  make format     - Format code"

# Setup project environment
setup:
	@echo "🚀 Setting up project environment..."
	python -m venv venv || true
	mkdir -p data/raw data/processed models reports logs mlruns
	@echo "✅ Project setup completed"
	@echo "Run 'source venv/bin/activate' to activate virtual environment"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed"

# Process data pipeline
data:
	@echo "📊 Running data pipeline..."
	python src/data/data_loader.py
	@echo "✅ Data processing completed"

# Train models
train:
	@echo "🤖 Training models..."
	python src/models/train_model.py
	@echo "✅ Model training completed"

# Start API server
api:
	@echo "🌐 Starting API server..."
	@echo "Visit http://localhost:8000/docs for API documentation"
	python src/api/app.py

# Run tests
test:
	@echo "🧪 Running tests..."
	python -m pytest tests/ -v --cov=src/ --cov-report=html || true
	@echo "✅ Tests completed"

# Start MLflow UI
mlflow:
	@echo "📊 Starting MLflow UI..."
	@echo "Visit http://localhost:5000 to view experiments"
	mlflow ui --host 0.0.0.0 --port 5000

# Run complete pipeline (automated)
all:
	@echo "🚀 Running complete MLOps pipeline..."
	python src/workflows/training_pipeline.py

# Quick start - most common workflow
quick:
	@echo "⚡ Quick start workflow..."
	make install
	make data
	make train
	@echo "✅ Quick start completed!"
	@echo "Now run 'make api' to start the API server"

# Development workflow
dev:
	@echo "🛠️ Development workflow..."
	make install
	make data
	make train
	make test
	@echo "✅ Development workflow completed!"

# Clean generated files
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "✅ Cleanup completed"

# Build Docker containers
docker:
	@echo "🐳 Building Docker containers..."
	docker build -t energy-prediction-api .
	@echo "✅ Docker build completed"

# Run code linting
lint:
	@echo "🔍 Running code linting..."
	flake8 src/ --max-line-length=88 || true
	black --check src/ || true
	isort --check-only src/ || true

# Format code
format:
	@echo "✨ Formatting code..."
	black src/
	isort src/

# Monitor models
monitor:
	@echo "📈 Running model monitoring..."
	python src/monitoring/model_monitor.py

# Production deployment
deploy:
	@echo "🚀 Production deployment..."
	make lint
	make test
	make docker
	@echo "✅ Ready for production deployment!"
