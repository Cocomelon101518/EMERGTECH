import pandas as pd
import numpy as np
import os
import json

def get_machine_statuses():
    """Reads the machine status from the JSON file."""
    if not os.path.exists('machine_status.json'):
        return {}
    with open('machine_status.json', 'r') as f:
        return json.load(f)

def generate_daily_data(day=1, num_records=25, save_to_disk=False):
    """
    Generates synthetic machine data for a specific day, respecting machine status.
    """
    if not os.path.exists('simulation_data'):
        os.makedirs('simulation_data')

    statuses = get_machine_statuses()

    # Base characteristics for consistency
    np.random.seed(42)
    base_temp = np.random.normal(70, 5, num_records)
    base_vibration = np.random.normal(0.5, 0.05, num_records)
    base_pressure = np.random.normal(50, 2, num_records)
    base_sound = np.random.normal(60, 3, num_records)
    base_hours = np.random.randint(50, 150, num_records)

    # Create the dataframe for the new day
    new_day_df = pd.DataFrame(columns=['Machine_ID', 'Temperature', 'Vibration', 'Pressure', 'Sound_Level', 'Hours_Used', 'Failure'])

    # Load previous day's data if it exists, to handle 'Not Operating' machines
    previous_day_df = None
    if day > 1:
        prev_day_path = os.path.join('simulation_data', f'day_{day-1}.csv')
        if os.path.exists(prev_day_path):
            previous_day_df = pd.read_csv(prev_day_path)

    for i in range(num_records):
        machine_id = f'M{i+1}'
        status = statuses.get(machine_id, 'Operating')

        machine_data = {}
        if status == 'Replaced':
            # Reset machine to a 'new' state
            machine_data = {
                'Temperature': np.random.normal(65, 2),
                'Vibration': np.random.normal(0.4, 0.02),
                'Pressure': np.random.normal(50, 1),
                'Sound_Level': np.random.normal(55, 2),
                'Hours_Used': np.random.randint(0, 10),
            }
        elif status == 'Not Operating' and previous_day_df is not None:
            # Carry over data from the previous day
            machine_data = previous_day_df[previous_day_df['Machine_ID'] == machine_id].iloc[0].to_dict()
        else: # 'Operating' or 'Repaired'
            np.random.seed(day * (i + 1))
            daily_hours_increase = np.random.randint(8, 16)
            hours_used = base_hours[i] + (day * daily_hours_increase)
            temp_increase = (day * np.random.normal(0.1, 0.05))
            vibration_increase = (day * np.random.normal(0.002, 0.001))
            sound_increase = (day * np.random.normal(0.05, 0.02))

            machine_data = {
                'Temperature': base_temp[i] + temp_increase,
                'Vibration': base_vibration[i] + vibration_increase,
                'Pressure': base_pressure[i],
                'Sound_Level': base_sound[i] + sound_increase,
                'Hours_Used': hours_used,
            }

        machine_data['Machine_ID'] = machine_id
        new_day_df.loc[i] = machine_data

    # Determine failure based on thresholds
    failure = (
        (new_day_df['Temperature'] > 85) |
        (new_day_df['Vibration'] > 0.7) |
        (new_day_df['Hours_Used'] > 450) |
        (new_day_df['Sound_Level'] > 70)
    ).astype(int)
    new_day_df['Failure'] = failure

    # On Day 1, ensure robust training data
    if day == 1:
        new_day_df.loc[0, ['Temperature', 'Vibration', 'Failure']] = [90, 0.8, 1]
        new_day_df.loc[1, ['Temperature', 'Vibration', 'Hours_Used', 'Sound_Level', 'Failure']] = [60, 0.4, 100, 55, 0]

    if save_to_disk:
        filepath = os.path.join('simulation_data', f'day_{day}.csv')
        new_day_df.to_csv(filepath, index=False)
        if day == 1:
            new_day_df.to_csv('machine_data.csv', index=False)

    return new_day_df

if __name__ == "__main__":
    generate_daily_data(day=1, save_to_disk=True)
    print("Generated and saved initial data for Day 1, respecting machine statuses.")