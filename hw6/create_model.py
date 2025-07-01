import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

# Create a simple model and vectorizer
dv = DictVectorizer()
model = LinearRegression()

# Fit with some dummy data
dummy_data = [
    {'PULocationID': '1', 'DOLocationID': '2'},
    {'PULocationID': '2', 'DOLocationID': '3'}
]
dummy_target = [10, 20]  # dummy duration values

# Fit the vectorizer and model
X = dv.fit_transform(dummy_data)
model.fit(X, dummy_target)

# Save the model
with open('model.bin', 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Model saved to model.bin") 