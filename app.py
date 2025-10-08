from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
from generate_data import generate_daily_data
from model_training import retrain_model_with_history, train_initial_model

app = Flask(__name__)
app.secret_key = 'a_very_secret_key_for_flask_sessions'

def load_model():
    """Loads the model from disk."""
    try:
        return joblib.load('model.pkl')
    except FileNotFoundError:
        return None

def get_recommendation(machine):
    """Generates a recommendation based on machine status."""
    prob = machine['Failure_Probability_Value']
    hours = machine['Hours_Used']

    if machine['Failure_Prediction'] == 1:
        if prob > 0.9:
            return "Critical failure risk. Immediate replacement recommended."
        elif prob > 0.75:
            return "High failure risk. Schedule replacement soon. Repair is likely not cost-effective."
        else:
            return "Failure predicted. Immediate inspection and maintenance required."
    else:
        if hours > 400:
            return "Machine has high usage but is in good condition. Monitor closely and consider proactive maintenance."
        elif prob > 0.3:
            return "Condition is acceptable, but failure risk is rising. Schedule a preventive check-up."
        else:
            return "Machine is in good condition. Continue normal operation and routine checks."

@app.route('/')
def index():
    if 'day' not in session:
        session['day'] = 1

    day = session['day']
    data_filepath = os.path.join('simulation_data', f'day_{day}.csv')

    # Generate data for the current day if it doesn't exist
    if not os.path.exists(data_filepath):
        generate_daily_data(day=day, save_to_disk=True)

    # Read the data for the current day
    data = pd.read_csv(data_filepath)

    model = load_model()
    results = []
    if model:
        features = ['Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used']

        if all(feature in data.columns for feature in features):
            predictions = model.predict(data[features])
            prediction_proba = model.predict_proba(data[features])

            data['Failure_Prediction'] = predictions
            data['Failure_Probability_Value'] = prediction_proba[:, 1]
            data['Failure_Probability'] = [f"{p:.2%}" for p in prediction_proba[:, 1]]
            data['Recommendation'] = data.apply(get_recommendation, axis=1)

            results = data.to_dict(orient='records')
        else:
            flash("The data file is missing required columns.", "danger")
    else:
        flash("Model not found. Please train the initial model by running model_training.py.", "danger")

    return render_template('index.html', results=results, day=day)

@app.route('/next_day')
def next_day():
    if 'day' in session:
        session['day'] += 1
    else:
        session['day'] = 1

    # Pre-generate data for the new day
    generate_daily_data(day=session['day'], save_to_disk=True)

    return redirect(url_for('index'))

@app.route('/report_issue/<machine_id>')
def report_issue(machine_id):
    """Triggers model re-training when a machine issue is reported."""
    flash(f"Issue reported for {machine_id}. Re-training model with all historical data to improve future predictions.", "info")

    if retrain_model_with_history():
        flash("Model successfully re-trained with updated data.", "success")
    else:
        flash("Could not re-train the model. No historical data found.", "warning")

    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    """Resets the simulation back to Day 1."""
    session['day'] = 1
    # For a true reset, we might clear simulation_data and retrain,
    # but for now, just going to day 1 is sufficient.
    flash("Simulation has been reset to Day 1.", "info")
    return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the initial model exists on startup
    if not os.path.exists('model.pkl'):
        train_initial_model()
    app.run(debug=True, port=5001)