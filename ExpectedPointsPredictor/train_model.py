import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Debugging: Start of script
print("Script has started running...")

# Load and filter data
url = 'https://drive.google.com/uc?id=11P2lTHCZk-Jkfh1LlB2L78QfyZl4SW4k&export=download'
print("Attempting to load data from URL...")

try:
    df = pd.read_csv(url)
    print(f"Data loaded successfully. Shape: {df.shape}")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()  # Exit script if data cannot be loaded

# Filter the relevant columns
try:
    filtered_df = df[['yardline_100', 'down', 'ydstogo', 'fixed_drive_result']]
    print(f"Filtered data shape: {filtered_df.shape}")
except Exception as e:
    print(f"Error filtering data: {e}")
    exit()

# Map fixed_drive_result to points
def map_drive_result(result):
    if result == 'Touchdown':
        return 7
    elif result == 'Field goal':
        return 3
    else:
        return 0

try:
    filtered_df['points'] = filtered_df['fixed_drive_result'].apply(map_drive_result)
    print("Mapped fixed_drive_result to points.")
except Exception as e:
    print(f"Error mapping drive results: {e}")
    exit()

# Prepare data for the model
try:
    data = filtered_df[['yardline_100', 'down', 'ydstogo', 'points']].dropna()
    print(f"Prepared data for training. Shape: {data.shape}")
except Exception as e:
    print(f"Error preparing data: {e}")
    exit()

# Define features and target
try:
    X = data[['yardline_100', 'down', 'ydstogo']]
    y = data['points']
    print("Features and target defined.")
except Exception as e:
    print(f"Error defining features and target: {e}")
    exit()

# Train/test split and model fitting
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error training model: {e}")
    exit()

# Save the trained model as model.pkl
print("Attempting to save the model as model.pkl...")
try:
    with open('model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print("Model saved successfully as model.pkl.")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()

# Debugging: End of script
print("Script completed successfully.")
