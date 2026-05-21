import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load the data
df = pd.read_csv('processed_train_FD001.csv')

# 2. Automatically find the dead flatliner columns
flatline_columns = [col for col in df.columns if df[col].std() == 0]

# 3. Create df_clean by dropping those columns (This fixes the error!)
df_clean = df.drop(columns=flatline_columns)

print(f"Useless columns dropped: {flatline_columns}")
print(f"Cleaned data shape: {df_clean.shape}")

# 4. Identify the features we actually want to scale
features_to_scale = [col for col in df_clean.columns if col not in ['engine_id', 'cycle', 'RUL']]

# 5. Initialize and apply the Min-Max Scaler
scaler = MinMaxScaler()
df_clean[features_to_scale] = scaler.fit_transform(df_clean[features_to_scale])

print("\n--- Scaling Complete! ---")
print(df_clean[['sensor2', 'sensor3', 'sensor4']].head())

# 6. Save your Day 2 progress
df_clean.to_csv('final_cleaned_train.csv', index=False)
print("\nSaved Day 2 progress to 'final_cleaned_train.csv'!")