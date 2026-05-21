import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
import joblib

# 1. Load our perfectly cleaned data from Day 2
df = pd.read_csv('final_cleaned_train.csv')

# 2. Separate Features (X) and Target (y)
X_columns = [col for col in df.columns if col not in ['engine_id', 'cycle', 'RUL']]
X = df[X_columns].values  
y = df['RUL'].values       

print("--- Day 5: Training Deep Learning Engine ---")
print(f"Feeding {X.shape[0]} flight rows with {X.shape[1]} sensor inputs to the Neural Network...\n")

# 3. Initialize the Multilayer Perceptron (MLP) Regressor
# We create 2 hidden layers: Layer 1 has 64 neurons, Layer 2 has 32 neurons.
# 'relu' handles non-linear activation; 'adam' is our advanced gradient descent optimizer.
nn_model = MLPRegressor(
    hidden_layer_sizes=(64, 32), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42,
    verbose=True  # This tells Python to print the system error live as it trains!
)

# 4. Train the Neural Network
print("Neurons are firing... training started:")
nn_model.fit(X, y)

# 5. Evaluate training accuracy
nn_predictions = nn_model.predict(X)
nn_rmse = np.sqrt(np.mean((nn_predictions - y) ** 2))

print("\n--- Neural Network Training Complete! ---")
print(f"Neural Network Training RMSE: {nn_rmse:.2f} cycles")

# 6. Freeze the model brain over the old pkl file
joblib.dump(nn_model, 'turbine_rf_model.pkl')
print("Success! Deep Learning brain saved to 'turbine_rf_model.pkl'")