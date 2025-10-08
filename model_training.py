import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import glob

def retrain_model_with_history():
    """
    Re-trains the model using all available historical data from the simulation.
    """
    data_files = glob.glob(os.path.join('simulation_data', 'day_*.csv'))

    if not data_files:
        print("No historical data found to retrain the model.")
        return False

    # Load all historical data
    df_list = [pd.read_csv(file) for file in data_files]
    full_df = pd.concat(df_list, ignore_index=True)

    print(f"Re-training model with {len(full_df)} records from {len(data_files)} days.")

    # Prepare the data
    features = ['Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used']
    target = 'Failure'

    X = full_df[features]
    y = full_df[target]

    # Train the model with all available data
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the updated model
    joblib.dump(model, 'model.pkl')
    print("Model successfully re-trained and saved as model.pkl")
    return True

def train_initial_model():
    """
    Trains the initial model using only the first day's data.
    """
    try:
        df = pd.read_csv('machine_data.csv')
    except FileNotFoundError:
        print("Initial data file (machine_data.csv) not found. Please generate it first.")
        return

    # Prepare the data
    features = ['Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used']
    target = 'Failure'

    X = df[features]
    y = df[target]

    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Save the initial model
    joblib.dump(model, 'model.pkl')
    print("Initial model training complete and model saved as model.pkl")

if __name__ == "__main__":
    # When run directly, it trains the initial model
    train_initial_model()