# MLOps Homework 4 - Deployments

## Q1. Standard Deviation of Predicted Duration

See [`starter.ipynb`](starter.ipynb) - Run for March 2023 data.

```python
std_pred = y_pred.std()
```

**Answer:** 6.24

## Q2. Output File Size

Code in [`starter.ipynb`](starter.ipynb):

```python
df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)
file_size = os.path.getsize(output_file) / (1024 * 1024)
```

**Answer:** 66M

## Q3. Convert Notebook to Script

```bash
jupyter nbconvert --to script starter.ipynb
```

## Q4. Pipenv Hash

```bash
pipenv install scikit-learn==1.6.1 pandas pyarrow requests
```

Check [`Pipfile.lock`](./mlops-hw4/Pipfile.lock) for first scikit-learn hash.

**Answer:** 0650e730afb87402baa88afbf31c07b84c98272622aaba002559b614600ca691

## Q5. Parametrized Script

See [`script.py`](script.py) - Run with:

```bash
python script.py --year 2023 --month 4
```

**Answer:** 14.29

## Q6. Docker Container

See [`dockerfile`](dockerfile) - Run with:

```bash
docker build -t mlops-hw4 .
docker run mlops-hw4
```

**Answer:** 0.19