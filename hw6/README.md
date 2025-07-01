# MLOps Homework 6 - Testing

## Files
- [`batch.py`](batch.py) - Main prediction service
- [`docker-compose.yaml`](docker-compose.yaml) - Localstack configuration
- [`tests/test_batch.py`](tests/test_batch.py) - Unit tests
- [`tests/integration_test.py`](tests/integration_test.py) - Integration tests

## Setup & Run

```bash
# Install dependencies
pip install pytest pandas scikit-learn boto3 pyarrow

# Start Localstack
docker-compose up -d

# Create test model
python create_model.py

# Run unit tests
python -m pytest tests/test_batch.py -v

# Run integration test
PYTHONPATH=. python tests/integration_test.py
```

## Homework Answers

### Q1. Refactoring
Check [`batch.py`](batch.py):
- Has `main()` function with year/month parameters
- `categorical` is parameter for `read_data`
- Has `if __name__ == "__main__"` block

### Q2. Installing pytest
Required files in [`tests/`](tests/):
- `__init__.py`
- `test_batch.py`

### Q3. First Unit Test
Run unit tests and check output:
```bash
python -m pytest tests/test_batch.py -v
```
Answer: 2 rows remain after filtering

### Q4. Localstack Option
The correct option for localstack commands is: `--endpoint-url`

### Q5 & Q6. File Size and Duration Sum
Run integration test and check output:
```bash
PYTHONPATH=. python tests/integration_test.py
```
- Q5: Look for "Input file size: X bytes"
- Q6: Look for "Sum of predicted durations: X"

## Expected Answers

1. Q1: Code is already properly refactored
2. Q2: `__init__.py` needed in tests directory
3. Q3: 2 rows remain after filtering
4. Q4: `--endpoint-url` is the correct option
5. Q5: File size should be around 3620 bytes
6. Q6: The sum will be shown in integration test output

## Troubleshooting

If you get Docker/Localstack errors:
```bash
# Check if Docker is running
sudo systemctl status docker

# If not running, start it
sudo systemctl start docker

# Add your user to docker group (then log out and back in)
sudo usermod -aG docker $USER
```

If you get AWS CLI errors:
```bash
# Install AWS CLI
sudo apt install awscli
```

No real AWS credentials are needed - we're using Localstack with dummy credentials. 