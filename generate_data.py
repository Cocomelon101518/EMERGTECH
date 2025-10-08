import pandas as pd
import numpy as np

# Generate synthetic data
def generate_synthetic_data(num_records=25):
    np.random.seed(42)
    data = {
        'Machine_ID': [f'M{i}' for i in range(1, num_records + 1)],
        'Temperature': np.random.normal(70, 10, num_records),
        'Vibration': np.random.normal(0.5, 0.1, num_records),
        'Pressure': np.random.normal(50, 5, num_records),
        'Sound_Level': np.random.normal(60, 5, num_records),
        'Hours_Used': np.random.randint(100, 500, num_records),
        'Failure': np.random.randint(0, 2, num_records)
    }
    df = pd.DataFrame(data)

    # Introduce some correlation for failures
    df.loc[df['Temperature'] > 85, 'Failure'] = 1
    df.loc[df['Vibration'] > 0.7, 'Failure'] = 1
    df.loc[df['Hours_Used'] > 450, 'Failure'] = 1
    df.loc[df['Sound_Level'] > 70, 'Failure'] = 1

    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    df.to_csv('machine_data.csv', index=False)
    print("Generated machine_data.csv")