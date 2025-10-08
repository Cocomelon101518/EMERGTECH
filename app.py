from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load the trained model
try:
    model = joblib.load('model.pkl')
except FileNotFoundError:
    # This is a fallback for development, in a real scenario the model must exist
    model = None
    print("Model not found. Please train the model first by running `model_training.py`.")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
            # Save the uploaded file temporarily
            filepath = os.path.join('uploads', file.filename)
            if not os.path.exists('uploads'):
                os.makedirs('uploads')
            file.save(filepath)

            # Process the data
            data = pd.read_csv(filepath)
            features = ['Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used']

            if not all(feature in data.columns for feature in features):
                # Handle missing columns
                return render_template('index.html', error=f"CSV must contain: {', '.join(features)}")

            if model:
                predictions = model.predict(data[features])
                prediction_proba = model.predict_proba(data[features])
                data['Failure_Prediction'] = predictions
                data['Failure_Probability'] = [f"{p:.2%}" for p in prediction_proba[:, 1]]
                results = data.to_dict(orient='records')
            else:
                results = None

            # Clean up the uploaded file
            os.remove(filepath)

            return render_template('index.html', results=results, columns=data.columns)

    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True, port=5001)