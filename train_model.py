import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load your dataset
df = pd.read_csv("real_estate_10k.csv")

# Define features and target
X = df[['sqft', 'beds', 'baths', 'age', 'lot_size', 'garage_spaces', 'distance_to_transit', 'school_rating', 'has_pool']]
y = df['price']

# Train model
print("Training model... (this may take a few seconds)")
model = LinearRegression()
model.fit(X, y)

# Save using joblib
joblib.dump(model, "model.pkl")
print("Model trained and saved as model.pkl!")