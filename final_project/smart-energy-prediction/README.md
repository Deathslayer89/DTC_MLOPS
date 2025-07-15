# Smart Energy Consumption Prediction - MLOps Project

## Problem Statement

### The Challenge
Household energy consumption is unpredictable and wasteful, leading to:
- **High electricity bills** - Average household spends $1,400+ annually on electricity
- **Energy waste** - 20-30% of home energy consumption is unnecessary
- **Grid instability** - Unpredictable demand patterns strain electrical grids
- **Carbon emissions** - Residential energy use accounts for 20% of global CO2 emissions

### The Solution
Build an **intelligent energy prediction system** that:
1. **Predicts hourly energy consumption** using historical smart meter data
2. **Identifies usage patterns** to optimize consumption timing
3. **Reduces energy waste** through predictive insights
4. **Provides real-time recommendations** for energy-efficient behavior

### Business Impact
- **15-20% energy savings** through optimized usage patterns
- **$500-2000 annual savings** per household
- **Peak load reduction** for utility companies
- **Lower carbon footprint** through efficient energy use

### Technical Approach
Deploy a **production-ready MLOps pipeline** that:
- Processes 2M+ smart meter measurements from 4+ years of data
- Trains ML models with experiment tracking and versioning
- Deploys scalable prediction API with real-time monitoring
- Detects data drift and performance degradation automatically

## Dataset

**Source**: Household Electric Power Consumption (UCI Repository)
- **Size**: 2,075,259 measurements over 47 months (Dec 2006 - Nov 2010)
- **Features**: Global active/reactive power, voltage, intensity, 3 sub-metering channels
- **Target**: Predict global active power consumption (kW)
- **Frequency**: 1-minute interval measurements

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Flow                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │     Data    │    │     ML      │    │    Model    │    │ Monitoring  │
    │  Pipeline   │───▶│  Training   │───▶│ Deployment  │───▶│ & Alerting  │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │                   │
           ▼                   ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Extract   │    │   MLflow    │    │   FastAPI   │    │ Evidently   │
    │  Transform  │    │ Experiment  │    │   + Docker  │    │ Data Drift  │
    │   Load      │    │  Tracking   │    │ Kubernetes  │    │ Detection   │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
           │                   │                   │                   │
           ▼                   ▼                   ▼                   ▼
    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │ Data Quality│    │Hyperparameter│    │AWS ECS/ALB  │    │   Alerts    │
    │ Validation  │    │Optimization  │    │   Fargate   │    │ & Reports   │
    │ Feature Eng │    │Model Registry│    │CI/CD Pipeline│    │ Dashboard   │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                      Infrastructure & Operations                                │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
    │   Terraform │    │   GitHub    │    │   Prefect   │    │   AWS       │
    │     IaC     │    │  Actions    │    │  Workflow   │    │  Services   │
    │             │    │   CI/CD     │    │Orchestration│    │             │
    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Quick Start

### Prerequisites
- Python 3.9+
- Docker
- AWS Account (for cloud deployment)

### Command Flow
```bash
# 1. Environment Setup
make setup              # Setup project environment
make install           # Install dependencies

# 2. Data Pipeline
make data              # Process raw data → processed data

# 3. Model Training
make train             # Train models with MLflow tracking

# 4. Testing
make test              # Run unit & integration tests

# 5. Deployment
make deploy            # Build Docker + start API

# 6. Monitoring
make monitor           # Generate monitoring reports
```

### Local Development
```bash
# Clone repository
git clone <your-repo-url>
cd smart-energy-prediction

# Quick start workflow
make install
make data
make train
make deploy

# Test locally
curl http://localhost:8000/health
```

### Testing
```bash
# Run all tests
make test

# Run specific tests
pytest tests/test_data_pipeline.py -v
pytest tests/test_api.py -v
```

## Project Structure

```
smart-energy-prediction/
├── data/                    # Raw and processed data
├── src/
│   ├── data/               # Data processing
│   ├── features/           # Feature engineering
│   ├── models/             # Model training
│   ├── api/                # FastAPI deployment
│   ├── monitoring/         # Model monitoring
│   └── workflows/          # Prefect workflows
├── tests/                  # Unit and integration tests
├── infrastructure/         # Terraform IaC
├── .github/workflows/      # CI/CD pipelines
└── docker/                 # Docker configurations
```

## MLOps Components

- **Experiment Tracking**: MLflow for model versioning and metrics
- **Workflow Orchestration**: Prefect for pipeline automation
- **Model Deployment**: FastAPI with Docker containerization
- **Monitoring**: Evidently for data drift detection
- **Infrastructure**: Terraform for AWS resource management
- **CI/CD**: GitHub Actions for automated testing and deployment

## AWS Cloud Deployment

### Infrastructure Setup
```bash
# Configure AWS credentials
aws configure

# Deploy infrastructure with Terraform
cd infrastructure/
terraform init
terraform apply -auto-approve

# Build and push Docker image
cd ..
export AWS_ACCOUNT_ID=319197019085
export ECR_REPO_URL=319197019085.dkr.ecr.us-east-1.amazonaws.com/energy-api

docker build -t energy-prediction-api .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $ECR_REPO_URL
docker tag energy-prediction-api:latest $ECR_REPO_URL:latest
docker push $ECR_REPO_URL:latest
```

### Production Testing
```bash
# Test production API
curl http://smart-energy-alb-795435159.us-east-1.elb.amazonaws.com/health

# Test prediction endpoint
curl -X POST http://smart-energy-alb-795435159.us-east-1.elb.amazonaws.com/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "year": 2023
  }'
```

## Model Performance

- **Algorithm**: XGBoost Regressor
- **Target Metric**: Mean Absolute Error (MAE)
- **Performance**: MAE < 0.3 kW
- **Features**: 7 energy consumption metrics + 10 engineered time features

## API Endpoints

### Health Check
```bash
GET /health
Response: {"status": "healthy"}
```

### Model Info
```bash
GET /model/info
Response: {"model_name": "XGBoost", "version": "1.0", "features": [...]}
```

### Prediction
```bash
POST /predict
Content-Type: application/json

{
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
  "year": 2023
}
```

## Monitoring

### Generate Monitoring Report
```bash
make monitor
```

### View Reports
- Data quality report: `reports/data_quality_report.html`
- Data drift report: `reports/data_drift_report.html`
- Model performance: `reports/model_performance_report.html`

## Production URLs

- **API Endpoint**: http://smart-energy-alb-795435159.us-east-1.elb.amazonaws.com
- **Health Check**: http://smart-energy-alb-795435159.us-east-1.elb.amazonaws.com/health
- **API Documentation**: http://smart-energy-alb-795435159.us-east-1.elb.amazonaws.com/docs

## Infrastructure

**AWS Resources Created:**
- S3 buckets for data and model storage
- ECR repository for Docker images
- ECS cluster with Fargate tasks
- Application Load Balancer
- VPC with public subnets
- Security groups and IAM roles
- CloudWatch logging
