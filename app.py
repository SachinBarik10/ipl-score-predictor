# app.py

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
    <style>
    .main {
        padding: 20px;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.title("🏏 IPL Score Predictor")
st.markdown("""
This app predicts the IPL cricket match scores using a machine learning model.  
Select the match parameters from the sidebar to get a prediction!
""")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load and prepare data
@st.cache_data
def load_data():
    try:
        ipl = pd.read_csv('ipl_data.csv')  # Updated filename
        return ipl
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Create and compile model
@st.cache_resource
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_shape,)),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(216, activation='relu'),
        keras.layers.Dense(1, activation='linear')
    ])
    huber_loss = tf.keras.losses.Huber(delta=1.0)
    model.compile(optimizer='adam', loss=huber_loss)
    return model

# Main data processing
with st.spinner('Loading and preparing data...'):
    if not st.session_state.data_loaded:
        ipl = load_data()
        if ipl is not None:
            df = ipl.drop(['date', 'runs', 'wickets', 'overs', 'runs_last_5', 
                           'wickets_last_5', 'mid', 'striker', 'non-striker'], axis=1)

            X = df.drop(['total'], axis=1)
            y = df['total']

            # Initialize encoders
            st.session_state.venue_encoder = LabelEncoder()
            st.session_state.batting_team_encoder = LabelEncoder()
            st.session_state.bowling_team_encoder = LabelEncoder()
            st.session_state.striker_encoder = LabelEncoder()
            st.session_state.bowler_encoder = LabelEncoder()

            # Fit and transform
            X['venue'] = st.session_state.venue_encoder.fit_transform(X['venue'])
            X['bat_team'] = st.session_state.batting_team_encoder.fit_transform(X['bat_team'])
            X['bowl_team'] = st.session_state.bowling_team_encoder.fit_transform(X['bowl_team'])
            X['batsman'] = st.session_state.striker_encoder.fit_transform(X['batsman'])
            X['bowler'] = st.session_state.bowler_encoder.fit_transform(X['bowler'])

            # Split and scale data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            st.session_state.scaler = MinMaxScaler()
            X_train_scaled = st.session_state.scaler.fit_transform(X_train)
            X_test_scaled = st.session_state.scaler.transform(X_test)

            # Create and train model
            if st.session_state.model is None:
                st.session_state.model = create_model(X_train_scaled.shape[1])
                with st.spinner('Training model... Please wait...'):
                    st.session_state.model.fit(
                        X_train_scaled, y_train,
                        epochs=50,
                        batch_size=64,
                        validation_data=(X_test_scaled, y_test),
                        verbose=0
                    )
            st.success("✅ Model trained and ready!")
            st.session_state.data_loaded = True
            st.session_state.df = df

# Sidebar inputs
st.sidebar.header("Match Parameters")

if st.session_state.data_loaded:
    venue = st.sidebar.selectbox('Select Venue:', st.session_state.df['venue'].unique().tolist())
    batting_team = st.sidebar.selectbox('Select Batting Team:', st.session_state.df['bat_team'].unique().tolist())
    bowling_team = st.sidebar.selectbox('Select Bowling Team:', st.session_state.df['bowl_team'].unique().tolist())
    striker = st.sidebar.selectbox('Select Striker:', st.session_state.df['batsman'].unique().tolist())
    bowler = st.sidebar.selectbox('Select Bowler:', st.session_state.df['bowler'].unique().tolist())

    predict_button = st.sidebar.button('Predict Score', use_container_width=True)

    # Main content area
    if predict_button:
        if batting_team == bowling_team:
            st.error("⚠️ Batting team and Bowling team cannot be the same!")
        else:
            try:
                # Prepare input data
                decoded_venue = st.session_state.venue_encoder.transform([venue])
                decoded_batting_team = st.session_state.batting_team_encoder.transform([batting_team])
                decoded_bowling_team = st.session_state.bowling_team_encoder.transform([bowling_team])
                decoded_striker = st.session_state.striker_encoder.transform([striker])
                decoded_bowler = st.session_state.bowler_encoder.transform([bowler])

                input_data = np.array([decoded_venue, decoded_batting_team, decoded_bowling_team,
                                       decoded_striker, decoded_bowler])
                input_data = input_data.reshape(1, 5)
                input_data = st.session_state.scaler.transform(input_data)

                # Make prediction
                predicted_score = st.session_state.model.predict(input_data, verbose=0)
                predicted_score = int(predicted_score[0, 0])

                # Display prediction
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown(f"""
                        <div style='padding: 20px; background-color: #f0f2f6; border-radius: 10px; text-align: center;'>
                            <h2 style='color: #0066cc;'>Predicted Score</h2>
                            <h1 style='color: #002b80; font-size: 48px;'>{predicted_score}</h1>
                            <p style='color: #666666;'>Estimated runs</p>
                        </div>
                        """, unsafe_allow_html=True)

                # Match details
                st.markdown("### Match Details")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    * **Venue**: {venue}
                    * **Batting Team**: {batting_team}
                    * **Striker**: {striker}
                    """)
                with col2:
                    st.markdown(f"""
                    * **Bowling Team**: {bowling_team}
                    * **Bowler**: {bowler}
                    """)

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

    # About model
    with st.expander("About the Model"):
        st.markdown("""
        This IPL Score Predictor uses a Deep Learning model with:
        - Neural Network with multiple dense layers
        - Huber loss function for robust predictions
        - Trained on historical IPL match data
        - Inputs: venue, teams, striker, bowler
        """)

else:
    st.error("Please wait while the data is being loaded...")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit • IPL Score Predictor")
