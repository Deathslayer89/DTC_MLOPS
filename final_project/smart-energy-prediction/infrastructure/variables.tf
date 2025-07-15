variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Prefix for resource names"
  type        = string
  default     = "smart-energy"
}

variable "ecr_repo_name" {
  description = "Name of the ECR repository for the API image"
  type        = string
  default     = "energy-api"
}

variable "environment" {
  description = "Environment name (e.g. dev, prod)"
  type        = string
  default     = "dev"
} 