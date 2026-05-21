import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Load our perfectly cleaned data from Day 2
df = pd.read_csv('final_cleaned_train.csv')

# 2. Separate Features (X) and Target (y)
# We drop engine_id, cycle, and RUL because they aren't sensor inputs
X_columns = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL']]

X = df[X_columns].values  # Converts the sensor table into a pure mathematical matrix
y = df['RUL'].values       # Converts the RUL column into a mathematical vector

print("--- Data Split for Math Engine ---")
print(f"Shape of X (Inputs): {X.shape}  -> (Rows of flights, Number of sensors)")
print(f"Shape of y (Answers): {y.shape} -> (Rows of flights,)")

# 3. Initialize our mixing board knobs (Weights and Bias) to zero
num_samples, num_features = X.shape  # num_samples is 20631, num_features is 19
weights = np.zeros(num_features)
bias = 0.0

# 4. Set Hyperparameters (The settings for our training loop)
learning_rate = 0.1  # How big of a nudge we give the sliders each step
epochs = 500         # How many times the model reviews the entire dataset

print("\nStarting Gradient Descent Math Engine...")

# 5. The Training Loop
for epoch in range(epochs):
    # Step A: Make a prediction for all 20,631 rows using current weights
    predictions = np.dot(X, weights) + bias
    
    # Step B: Calculate the error (How far off were we?)
    error = predictions - y
    
    # Step C: Calculate the "Gradient" (Which direction to move the sliders)
    dw = (2 / num_samples) * np.dot(X.T, error)
    db = (2 / num_samples) * np.sum(error)
    
    # Step D: Nudge the weights and bias in the opposite direction of the error
    weights -= learning_rate * dw
    bias -= learning_rate * db
    
    # Step E: Every 50 iterations, print the average error (Mean Squared Error)
    if epoch % 50 == 0:
        mse = np.mean(error ** 2)
        print(f"Epoch {epoch}/{epochs} -> Average System Error (MSE): {mse:.2f}")

print("\n--- Training Complete! ---")
print("Optimized Master Knob (Bias):", bias)
print("First 3 Optimized Sensor Weights:", weights[:3])

# 6. Evaluate accuracy using Root Mean Squared Error (RMSE)
# Re-calculate final predictions using our newly optimized weights and bias
final_predictions = np.dot(X, weights) + bias

# Calculate the final differences, square them, find the mean, and take the square root
final_mse = np.mean((final_predictions - y) ** 2)
rmse = np.sqrt(final_mse)

print(f"Final Model Accuracy (RMSE): {rmse:.2f} cycles")
print("This means on average, our model's guess is off by about this many flights.")


print("\n--- Training Tuned Random Forest Regressor ---")

# Aggressive overfitting reduction:
# - Lower max_depth: prevents deep memorization
# - Higher min_samples_split: forces broader patterns
# - Add min_samples_leaf: ensures leaf nodes have enough data
# - Limit max_features: reduces feature correlation overfitting
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=8,              # Reduced from 15
    min_samples_split=30,      # Increased from 10
    min_samples_leaf=10,       # New: was default 1
    max_features='sqrt',       # New: limits features per split
    random_state=42
)

# Train the tuned model
rf_model.fit(X, y)

# Evaluate on training data
rf_predictions = rf_model.predict(X)
rf_rmse = np.sqrt(np.mean((rf_predictions - y) ** 2))
print(f"Tuned Training RMSE (Will be higher, which is good!): {rf_rmse:.2f} cycles")

# Save the newly tuned model over the old one
joblib.dump(rf_model, 'turbine_rf_model.pkl')
print("Success! Optimized 'turbine_rf_model.pkl' saved.")