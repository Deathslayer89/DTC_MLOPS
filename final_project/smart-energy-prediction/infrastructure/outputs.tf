output "ecr_repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.api_repo.repository_url
}

output "artifact_bucket_name" {
  description = "Name of the S3 bucket for artifacts"
  value       = aws_s3_bucket.artifact_bucket.id
}

output "s3_model_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.model_artifacts.id
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.cluster.name
}

output "alb_dns_name" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "subnet_ids" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "security_group_id" {
  description = "Security group ID for ECS tasks"
  value       = aws_security_group.ecs_tasks.id
}

output "target_group_arn" {
  description = "Target group ARN"
  value       = aws_lb_target_group.app.arn
}

output "random_suffix" {
  description = "Random suffix for unique resource names"
  value       = random_string.suffix.result
} 