# Streamlit Lottery Mutation Optimizer App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import random
import requests
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# API Endpoint and CSV File Name
API_URL = "https://data.ny.gov/resource/6nbc-h7bj.json"
CSV_FILE = "lottery_data.csv"

# Function to fetch data from API and store in CSV
@st.cache_data
def fetch_and_store_data():
    try:
        response = requests.get(API_URL)
        data = response.json()
        df = pd.DataFrame(data)
        # Extract relevant columns
        df = df[['draw_date', 'winning_numbers', 'bonus']]
        df['draw_date'] = pd.to_datetime(df['draw_date'])
        df = df.sort_values(by='draw_date', ascending=True)
        # Split winning numbers into separate columns
        df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6']] = df['winning_numbers'].str.split(" ", expand=True).astype(int)
        df['bonus'] = df['bonus'].astype(int)
        # Save to CSV
        df.to_csv(CSV_FILE, index=False)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load data
df = fetch_and_store_data()
if df.empty:
    st.stop()

# Sidebar settings
st.sidebar.header("Mutation Adjustment")
mutation_level = st.sidebar.slider("Mutation Strength", 0.0, 1.0, 0.5)

st.sidebar.subheader("XGBoost Filtering")
use_xgboost = st.sidebar.checkbox("Enable XGBoost Filtering", True)

# Train XGBoost classifier
def train_xgboost(df):
    # Use historical draws as positive examples
    pos = df[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']].astype(int).copy()
    pos = pos.reset_index(drop=True)
    pos['label'] = 1

    # Synthesize negative examples by generating random sequences (same shape)
    n_neg = len(pos)
    neg_values = np.random.randint(1, 60, size=(n_neg, 7))  # numbers 1..59 inclusive
    neg = pd.DataFrame(neg_values, columns=['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus'])
    neg['label'] = 0

    # Combine and shuffle
    data = pd.concat([pos, neg], ignore_index=True)
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    X = data[['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'bonus']].astype(int)
    y = data['label'].astype(int)

    # XGBoost: avoid use_label_encoder warning and set eval_metric
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)
    model.fit(X, y)
    return model

if use_xgboost:
    xgb_model = train_xgboost(df)

# Generate random mutations (including the bonus in sequence)
def generate_mutations(draw, bonus, mutation_level):
    full_sequence = draw + [bonus]
    mutated_sequence = full_sequence.copy()
    # Number of mutations proportional to mutation_level
    n_mut = max(1, int(round(mutation_level * len(full_sequence))))
    for _ in range(n_mut):
        idx = random.randint(0, len(full_sequence) - 1)
        mutated_sequence[idx] = random.randint(1, 59)
    # Keep sequence order (do not sort) so transitions make sense
    return mutated_sequence

# LSTM model for sequence learning (not trained, just a placeholder)
def build_lstm():
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(7, 1)),
        Dense(7, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

lstm_model = build_lstm()

# Show generated predictions
st.subheader("Generated Predictions")
selected_row = df.iloc[-1]
selected_draw = [int(selected_row[f'num{i}']) for i in range(1, 7)]
selected_bonus = int(selected_row['bonus'])

st.write("Base Draw:", selected_draw, "Bonus:", selected_bonus)
mutated_sequences = [generate_mutations(selected_draw, selected_bonus, mutation_level) for _ in range(10)]

# Filter using XGBoost
if use_xgboost:
    # Ensure predictor input shape is correct: each seq must be length 7
    filtered_sequences = []
    for seq in mutated_sequences:
        pred = xgb_model.predict([seq])[0]
        if int(pred) == 1:
            filtered_sequences.append(seq)
else:
    filtered_sequences = mutated_sequences

# Display results
st.write("Filtered Predictions:")
for seq in filtered_sequences:
    main_numbers, bonus_number = seq[:-1], seq[-1]
    st.write(f"Numbers: {main_numbers}, Bonus: {bonus_number}")

# Transition Graph
st.subheader("Transition Graph")
G = nx.DiGraph()
for seq in filtered_sequences:
    for i in range(len(seq) - 1):
        G.add_edge(seq[i], seq[i + 1])

plt.figure(figsize=(8, 6))
nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray')
st.pyplot(plt)
plt.close()

# Heatmap
st.subheader("Number Transition Heatmap")
transition_matrix = np.zeros((59, 59))
for seq in filtered_sequences:
    for i in range(len(seq) - 1):
        transition_matrix[seq[i] - 1, seq[i + 1] - 1] += 1

plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, cmap="coolwarm", linewidths=0.5)
st.pyplot(plt)
plt.close()

st.write("Mutation optimization completed!")
