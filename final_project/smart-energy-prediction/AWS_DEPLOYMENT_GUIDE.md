
# 🚀 AWS Deployment Guide - Step by Step

## Prerequisites ✅

1. **AWS Account** with programmatic access
2. **AWS CLI** installed and configured
3. **Terraform** installed (version >= 1.3.0)
4. **Docker** installed and running
5. **Git** for version control

## Step 1: Setup AWS Credentials

```bash
# Install AWS CLI (if not installed)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS credentials
aws configure
# Enter your:
# - AWS Access Key ID
# - AWS Secret Access Key
# - Default region (us-east-1)
# - Default output format (json)

# Verify configuration
aws sts get-caller-identity
```

## Step 2: Install Terraform

```bash
# Download and install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Verify installation
terraform version
```

## Step 3: Prepare Local Environment

```bash
# Activate conda environment
conda activate base

# Install Python dependencies
pip install -r requirements.txt

# Process data and train model
make data
make train
```

## Step 4: Deploy AWS Infrastructure

```bash
# Navigate to infrastructure directory
cd infrastructure/

# Initialize Terraform
terraform init

# Review what will be created
terraform plan

# Deploy infrastructure
terraform apply -auto-approve

# Note down the outputs (important!)
terraform output
```

### What Gets Created:
- **S3 Buckets**: For data and model artifacts
- **ECR Repository**: For Docker images
- **ECS Cluster**: For running containers
- **VPC & Networking**: Subnets, security groups, load balancer
- **IAM Roles**: For ECS task execution
- **CloudWatch**: For logging and monitoring

## Step 5: Build and Push Docker Image

```bash
# Go back to project root
cd ..

# Get AWS account ID
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-east-1

# Build Docker image
docker build -t energy-prediction-api .

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Tag image for ECR
docker tag energy-prediction-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/energy-api:latest

# Push to ECR
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/energy-api:latest
```

## Step 6: Deploy ECS Service

```bash
# Create ECS service using the task definition
aws ecs create-service \
  --cluster smart-energy-cluster \
  --service-name energy-api-service \
  --task-definition smart-energy-task \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx,subnet-yyy],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:us-east-1:xxx:targetgroup/smart-energy-tg/xxx,containerName=api,containerPort=8000"

# Check service status
aws ecs describe-services --cluster smart-energy-cluster --services energy-api-service
```

## Step 7: Test Deployment

```bash
# Get the load balancer DNS name
ALB_DNS=$(terraform output -raw alb_dns_name)

# Test health endpoint
curl http://$ALB_DNS/health

# Test prediction endpoint
curl -X POST http://$ALB_DNS/predict \
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

## Step 8: Setup Monitoring

```bash
# Upload monitoring data to S3
aws s3 cp reports/ s3://smart-energy-artifacts-$(terraform output -raw random_suffix)/reports/ --recursive

# Run monitoring pipeline
make monitor
```

## Step 9: Configure GitHub Actions (Optional)

```bash
# Add these secrets to your GitHub repository:
# Go to Settings → Secrets and variables → Actions

# Add these secrets:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_ACCOUNT_ID
# - AWS_REGION
```

## Step 10: Final Verification

```bash
# Check all AWS resources
echo "=== ECS Cluster ==="
aws ecs describe-clusters --clusters smart-energy-cluster

echo "=== ECR Repository ==="
aws ecr describe-repositories --repository-names energy-api

echo "=== S3 Buckets ==="
aws s3 ls | grep smart-energy

echo "=== Load Balancer ==="
aws elbv2 describe-load-balancers --names smart-energy-alb

echo "=== Test API ==="
curl http://$(terraform output -raw alb_dns_name)/health
```

## 🎉 Submission Ready!

Your MLOps project is now deployed to AWS! 

### What to Submit:
1. **GitHub Repository URL** containing:
   - Source code
   - Infrastructure code (Terraform)
   - Documentation (README.md)
   - CI/CD configuration

2. **Live Demo URL**: `http://$(terraform output -raw alb_dns_name)`

### Clean Up (After Submission)

```bash
# Destroy infrastructure to avoid costs
terraform destroy -auto-approve

# Delete ECR images
aws ecr batch-delete-image --repository-name energy-api --image-ids imageTag=latest
```

## Troubleshooting

### Common Issues:

1. **Terraform Apply Fails**: 
   - Check AWS credentials: `aws sts get-caller-identity`
   - Verify region: `aws configure get region`

2. **Docker Push Fails**:
   - Re-authenticate: `aws ecr get-login-password | docker login...`
   - Check repository exists: `aws ecr describe-repositories`

3. **ECS Service Won't Start**:
   - Check task definition: `aws ecs describe-task-definition --task-definition smart-energy-task`
   - Check logs: `aws logs describe-log-groups`

4. **API Not Responding**:
   - Check ECS service: `aws ecs describe-services --cluster smart-energy-cluster --services energy-api-service`
   - Check target group health: `aws elbv2 describe-target-health --target-group-arn <arn>`

### Support Commands:

```bash
# View ECS logs
aws logs get-log-events --log-group-name /ecs/smart-energy --log-stream-name <stream-name>

# Check ECS task status
aws ecs list-tasks --cluster smart-energy-cluster
aws ecs describe-tasks --cluster smart-energy-cluster --tasks <task-arn>

# Monitor load balancer
aws elbv2 describe-target-health --target-group-arn <target-group-arn>
```

**🚀 Your MLOps project is now production-ready on AWS!**