import pandas as pd

# 1. Create a list of meaningful names for all 26 columns
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

# 2. Read the file again, but this time pass our names list into the 'names' parameter
df = pd.read_csv("train_FD001.txt", sep=r'\s+', header=None, names=column_names)

# 3. Print the head again to see the gorgeous clean columns
print("--- Columns labeled successfully! ---")
print(df[['engine_id', 'cycle', 'sensor1', 'sensor2']].head())

# 4. Find the maximum cycle for each engine group
# This looks at all rows for an engine, finds the max cycle, and repeats it for every row
max_cycles = df.groupby('engine_id')['cycle'].transform('max')

# 5. Calculate RUL using our formula: Max Cycle - Current Cycle
df['RUL'] = max_cycles - df['cycle']

print("\n--- RUL Column Calculated! ---")
print(df[['engine_id', 'cycle', 'RUL']].head(15))

# 6. Save the processed dataframe to a new CSV file
df.to_csv('processed_train_FD001.csv', index=False)
print("\nSuccess! 'processed_train_FD001.csv' has been saved to your folder.")