import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv('machine_data.csv')

# Prepare the data
features = ['Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used']
target = 'Failure'

X = df[features]
y = df[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'model.pkl')

print("Model training complete and model saved as model.pkl")