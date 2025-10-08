from flask import Flask, render_template, request, redirect, url_for, session, flash
import pandas as pd
import joblib
import os
import json
from generate_data import generate_daily_data, get_machine_statuses
from model_training import retrain_model_with_history, train_initial_model

app = Flask(__name__)
app.secret_key = 'a_very_secret_key_for_flask_sessions_v2'

def load_model():
    """Loads the model from disk."""
    try:
        return joblib.load('model.pkl')
    except FileNotFoundError:
        return None

def update_machine_status(machine_id, new_status):
    """Updates the status of a specific machine in the JSON file."""
    statuses = get_machine_statuses()
    statuses[machine_id] = new_status
    with open('machine_status.json', 'w') as f:
        json.dump(statuses, f, indent=4)

def get_recommendation(machine):
    """Generates a recommendation based on machine status."""
    prob = machine.get('Failure_Probability_Value', 0)
    hours = machine.get('Hours_Used', 0)
    status = machine.get('Status', 'Operating')

    if status == 'Not Operating':
        return "This machine is not operating. Choose to mark it as repaired or replaced."
    if machine.get('Failure_Prediction') == 1:
        if prob > 0.9:
            return "Critical failure risk. Immediate replacement is strongly recommended."
        elif prob > 0.75:
            return "High failure risk. Schedule replacement soon. Repair is not likely to be cost-effective."
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

    if not os.path.exists(data_filepath):
        generate_daily_data(day=day, save_to_disk=True)

    data = pd.read_csv(data_filepath)
    statuses = get_machine_statuses()

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
            data['Status'] = data['Machine_ID'].map(statuses)
            data['Recommendation'] = data.apply(get_recommendation, axis=1)

            results = data.to_dict(orient='records')
        else:
            flash("The data file is missing required columns.", "danger")
    else:
        flash("Model not found. Please train the initial model.", "danger")

    return render_template('index.html', results=results, day=day)

@app.route('/next_day')
def next_day():
    session['day'] = session.get('day', 1) + 1
    generate_daily_data(day=session['day'], save_to_disk=True)
    return redirect(url_for('index'))

@app.route('/report_issue/<machine_id>')
def report_issue(machine_id):
    update_machine_status(machine_id, 'Not Operating')
    flash(f"{machine_id} has been marked as 'Not Operating'.", "warning")
    return redirect(url_for('index'))

@app.route('/resolve_issue/<machine_id>/<action>')
def resolve_issue(machine_id, action):
    if action not in ['Repaired', 'Replaced']:
        flash("Invalid action.", "danger")
        return redirect(url_for('index'))

    update_machine_status(machine_id, action)
    flash(f"{machine_id} has been marked as '{action}'. Re-training model with new data.", "info")

    if retrain_model_with_history():
        flash("Model successfully re-trained with updated historical data.", "success")
    else:
        flash("Could not re-train model. No historical data found.", "warning")

    return redirect(url_for('index'))

@app.route('/reset')
def reset():
    """Resets the simulation and machine statuses."""
    session['day'] = 1

    # Reset statuses
    num_records = len(get_machine_statuses())
    statuses = {f'M{i+1}': 'Operating' for i in range(num_records)}
    with open('machine_status.json', 'w') as f:
        json.dump(statuses, f, indent=4)

    # Clear old simulation data and regenerate day 1
    for f in os.listdir('simulation_data'):
        os.remove(os.path.join('simulation_data', f))
    generate_daily_data(day=1, save_to_disk=True)

    # Retrain the initial model
    train_initial_model()

    flash("Simulation has been completely reset to Day 1.", "info")
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('model.pkl'):
        train_initial_model()
    app.run(debug=True, port=5001)