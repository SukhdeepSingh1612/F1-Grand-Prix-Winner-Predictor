import streamlit as st
import pandas as pd
import numpy as np
import random

# Load your trained model and encoders
import joblib

# Load model and encoders
model = joblib.load('xgb_model.pkl')   # make sure you saved your model
le_driver = joblib.load('le_driver.pkl')
le_constructor = joblib.load('le_constructor.pkl')
le_race = joblib.load('le_race.pkl')

# Define driver-constructor mapping
# Full list of 20 drivers and their constructors
all_drivers = [
    'Max Verstappen', 'Sergio Pérez',           # Red Bull
    'Lewis Hamilton', 'George Russell',          # Mercedes
    'Charles Leclerc', 'Carlos Sainz',            # Ferrari
    'Lando Norris', 'Oscar Piastri',              # McLaren
    'Fernando Alonso', 'Lance Stroll',            # Aston Martin
    'Esteban Ocon', 'Pierre Gasly',               # Alpine F1 Team
    'Yuki Tsunoda', 'Nyck de Vries',           # RB (formerly AlphaTauri)
    'Valtteri Bottas', 'Guanyu Zhou',             # Sauber (formerly Alfa Romeo)
    'Kevin Magnussen', 'Nico Hülkenberg',         # Haas
    'Alexander Albon', 'Logan Sargeant'           # Williams
]

# Matching constructors
constructor_map = {
    'Max Verstappen': 'Red Bull',
    'Sergio Pérez': 'Red Bull',
    'Lewis Hamilton': 'Mercedes',
    'George Russell': 'Mercedes',
    'Charles Leclerc': 'Ferrari',
    'Carlos Sainz': 'Ferrari',
    'Lando Norris': 'McLaren',
    'Oscar Piastri': 'McLaren',
    'Fernando Alonso': 'Aston Martin',
    'Lance Stroll': 'Aston Martin',
    'Esteban Ocon': 'Alpine F1 Team',
    'Pierre Gasly': 'Alpine F1 Team',
    'Yuki Tsunoda': 'AlphaTauri',  # Red Bull's sister team (formerly AlphaTauri)
    'Nyck de Vries': 'AlphaTauri',
    'Valtteri Bottas': 'Alfa Romeo',
    'Guanyu Zhou': 'Alfa Romeo',
    'Kevin Magnussen': 'Haas F1 Team',
    'Nico Hülkenberg': 'Haas F1 Team',
    'Alexander Albon': 'Williams',
    'Logan Sargeant': 'Williams'
}



# Streamlit app
st.title("🏎️ F1 Race Winner Predictor")

# Select Race
race_name = st.text_input('Enter Race Name (Example: Bahrain Grand Prix)', 'Bahrain Grand Prix')

# Option to randomize grid
random_grid = st.checkbox('Randomize Grid Positions', value=True)

# Prepare race data
race_data = []

for driver in all_drivers:
    grid_position = random.randint(1, 20) if random_grid else st.number_input(f'Grid Position for {driver}', 1, 20, 10)
    
    try:
        driver_encoded = le_driver.transform([driver])[0]
        constructor_encoded = le_constructor.transform([constructor_map[driver]])[0]
        race_encoded = le_race.transform([race_name])[0]

        race_data.append({
            'Driver': driver,
            'Grid': grid_position,
            'Driver_Encoded': driver_encoded,
            'Constructor_Encoded': constructor_encoded,
            'Race_Encoded': race_encoded,
            'Finished': 1
        })
    except Exception as e:
        st.error(f"Encoding error: {e}")

# Predict button
if st.button('Predict Winner'):
    new_race_df = pd.DataFrame(race_data)

    # Define features as per your model
    features = ['Grid', 'Driver_Encoded', 'Constructor_Encoded', 'Race_Encoded', 'Finished']

    # Predict probabilities
    winner_prob = model.predict_proba(new_race_df[features])

    # Add probabilities
    new_race_df['Win_Probability'] = winner_prob[:, 1]

    # Find the winner
    predicted_winner = new_race_df.loc[new_race_df['Win_Probability'].idxmax()]

    st.success(f"🏆 Predicted Winner: **{predicted_winner['Driver']}** with winning probability of {predicted_winner['Win_Probability']:.4f}")

    st.subheader("📊 All Drivers' Winning Probabilities")
    st.dataframe(new_race_df[['Driver', 'Grid', 'Win_Probability']].sort_values(by='Win_Probability', ascending=False))
