# MLOps Homework 3 - NYC Taxi Training pipelines

## Questions & Answers

**Question 1. Select the Tool**  
What's the name of the orchestrator you chose?  
**Answer: MLflow**

**Question 2. Version**  
What's the version of the orchestrator?  
**Answer: 2.22.0**

**Question 3. Creating a pipeline**  
How many records did we load?
- 3,003,766
- 3,203,766  
- 3,403,766
- 3,603,766  
**Answer: 3,403,766**

**Question 4. Data preparation**  
What's the size of the result after data preparation?
- 2,903,766
- 3,103,766
- 3,316,216
- 3,503,766  
**Answer: 3,316,216**

**Question 5. Train a model**  
What's the intercept of the model?
- 21.77
- 24.77
- 27.77
- 31.77  
**Answer: 24.77**

**Question 6. Register the model**  
What's the size of the model? (`model_size_bytes` field):
- 14,534
- 9,534
- 4,534
- 1,534  
**Answer: 4,534** (Actual value: 4501, closest option is 4,534)

## Setup

1. Activate conda environment: `conda activate exp-tracking-env`
2. Run training: `python3 train.py`
3. Start MLflow UI: `mlflow ui`
4. Access UI at: `http://localhost:5000` 