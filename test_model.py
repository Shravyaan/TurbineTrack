import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

print("--- Day 5: Running Fixed Blind Test Pipeline ---")

# 1. Load Training Data to grab the EXACT correct scaling metrics
train_df = pd.read_csv('processed_train_FD001.csv')
flatline_cols = ['setting3', 'sensor1', 'sensor5', 'sensor10', 'sensor16']
train_df_clean = train_df.drop(columns=flatline_cols)
features_to_scale = [col for col in train_df_clean.columns if col not in ['engine_id', 'cycle', 'RUL']]

# Fit the scaler ONLY on training data
scaler = MinMaxScaler()
scaler.fit(train_df_clean[features_to_scale])

# 2. Load the raw blind test data
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
test_df = pd.read_csv("test_FD001.txt", sep=r'\s+', header=None, names=column_names)
test_df_clean = test_df.drop(columns=flatline_cols)

# 3. TRANSFORM the test features (DO NOT FIT!)
# This uses the exact training minimums and maximums
test_df_clean[features_to_scale] = scaler.transform(test_df_clean[features_to_scale])

# 4. Grab ONLY the last recorded row for each engine
test_last_cycles = test_df_clean.groupby('engine_id').last().reset_index()
X_test = test_last_cycles[features_to_scale].values

# 5. Load our frozen Random Forest model
model = joblib.load('turbine_rf_model.pkl')

# 6. Make our blind predictions
predicted_rul = model.predict(X_test)

# 7. Load NASA's secret true answers
true_rul = pd.read_csv("RUL_FD001.txt", header=None)[0].values

# 8. Calculate our true Blind Test Error (RMSE)
test_mse = np.mean((predicted_rul - true_rul) ** 2)
test_rmse = np.sqrt(test_mse)

print(f"\nTarget Evaluation Complete!")
print(f"Fixed Blind Test Dataset Model Accuracy (RMSE): {test_rmse:.2f} cycles")