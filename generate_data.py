import pandas as pd
import numpy as np
import os

def generate_daily_data(day=1, num_records=25, save_to_disk=False):
    """
    Generates synthetic machine data for a specific day and optionally saves it.
    """
    # Create directory if it doesn't exist
    if not os.path.exists('simulation_data'):
        os.makedirs('simulation_data')

    # Consistent base characteristics
    np.random.seed(42)
    base_temp = np.random.normal(70, 5, num_records)
    base_vibration = np.random.normal(0.5, 0.05, num_records)
    base_pressure = np.random.normal(50, 2, num_records)
    base_sound = np.random.normal(60, 3, num_records)
    base_hours = np.random.randint(50, 150, num_records)

    # Daily variations
    np.random.seed(day)
    daily_hours_increase = np.random.randint(8, 16, num_records)
    hours_used = base_hours + (day * daily_hours_increase)

    temp_increase = (day * np.random.normal(0.1, 0.05, num_records))
    vibration_increase = (day * np.random.normal(0.002, 0.001, num_records))
    sound_increase = (day * np.random.normal(0.05, 0.02, num_records))

    data = {
        'Machine_ID': [f'M{i}' for i in range(1, num_records + 1)],
        'Temperature': base_temp + temp_increase,
        'Vibration': base_vibration + vibration_increase,
        'Pressure': base_pressure,
        'Sound_Level': base_sound + sound_increase,
        'Hours_Used': hours_used,
    }
    df = pd.DataFrame(data)

    # Determine failure
    failure = (
        (df['Temperature'] > 85) |
        (df['Vibration'] > 0.7) |
        (df['Hours_Used'] > 450) |
        (df['Sound_Level'] > 70)
    ).astype(int)
    df['Failure'] = failure

    if save_to_disk:
        filepath = os.path.join('simulation_data', f'day_{day}.csv')
        df.to_csv(filepath, index=False)
        # Also update the main machine_data.csv for the initial model training
        if day == 1:
            df.to_csv('machine_data.csv', index=False)

    return df

if __name__ == "__main__":
    # Generate data for day 1 and save it
    generate_daily_data(day=1, save_to_disk=True)
    print("Generated and saved initial data for Day 1.")