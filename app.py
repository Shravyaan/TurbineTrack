import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

# ====================================================================
# 1. INITIALIZATION & DATA PIPELINE CALIBRATION
# ====================================================================

# Load your optimized champion Random Forest model brain
model = joblib.load('turbine_rf_model.pkl')

# Define the exact 5 flatline sensors we dropped on Day 2
flatline_cols = ['setting3', 'sensor1', 'sensor5', 'sensor10', 'sensor16']

# Load training data metrics to set up the EXACT same mathematical scale
train_df = pd.read_csv('processed_train_FD001.csv')
train_df_clean = train_df.drop(columns=flatline_cols)
features_to_scale = [col for col in train_df_clean.columns if col not in ['engine_id', 'cycle', 'RUL']]

# Fit our master scaler so the UI speaks the exact same language as the model
scaler = MinMaxScaler()
scaler.fit(train_df_clean[features_to_scale])

# ====================================================================
# 2. STREAMLIT UI LAYOUT ARCHITECTURE
# ====================================================================

# Configure the browser tab title and set the layout to widescreen
st.set_page_config(page_title="TurbineTrack AI", layout="wide")

# Main Dashboard Header
st.title("✈️ TurbineTrack: Predictive Maintenance Dashboard")
st.markdown("---")
st.markdown("### Real-Time Aircraft Fleet Telemetry Monitor")
st.write("Upload raw engine sensor telemetry below to instantly predict Remaining Useful Life (RUL) using our optimized Random Forest Engine.")

# Setup the dark control sidebar on the left
st.sidebar.header("🕹️ Deployment Controls")
uploaded_file = st.sidebar.file_uploader("Upload Telemetry File (test_FD001.txt)", type=["txt", "csv"])

# ====================================================================
# 3. LIVE ENGINE HEALTH COMPUTATION PIPELINE
# ====================================================================

# Check if the user has dropped a file into the upload box
if uploaded_file is not None:
    # Read the raw telemetry data stream
    column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    df = pd.read_csv(uploaded_file, sep=r'\s+', header=None, names=column_names)
    
    # Run the exact cleaning and scaling steps in real-time
    df_clean = df.drop(columns=flatline_cols)
    df_clean[features_to_scale] = scaler.transform(df_clean[features_to_scale])
    
    # ISOLATE MEMORY LOGIC: Grab only the very last flight row for every single engine
    latest_states = df_clean.groupby('engine_id').last().reset_index()
    X_live = latest_states[features_to_scale].values
    
    # Run the live data through our frozen machine learning model
    live_predictions = model.predict(X_live)
    
    # Add the predictions back into our tracking matrix
    latest_states['Predicted_RUL'] = np.round(live_predictions).astype(int)
    
    # ----------------------------------------------------------------
    # 4. FRONTEND METRIC CARDS DISPLAY
    # ----------------------------------------------------------------
    st.subheader("📊 Featured Fleet Asset Health")
    
    # Isolate Engine #1 from the uploaded batch to display as a main KPI showcase
    featured_engine = latest_states.iloc[0]
    predicted_life = int(featured_engine['Predicted_RUL'])
    
    # Create 3 clean, side-by-side visual columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label=f"Engine #{int(featured_engine['engine_id'])} Total Cycles Flown", 
            value=f"{int(featured_engine['cycle'])} Flights"
        )
        
    with col2:
        st.metric(
            label="Predicted Remaining Useful Life (RUL)", 
            value=f"{predicted_life} Cycles Left"
        )
        
    with col3:
        # Dynamic color-coded alert banner based on safety windows
        if predicted_life > 50:
            st.success("🟢 STATUS: NOMINAL (Engine Healthy)")
        elif predicted_life > 20:
            st.warning("⚠️ STATUS: MAINTENANCE WARNING (Order Parts)")
        else:
            st.error("🚨 STATUS: CRITICAL FAILURE RISK (Ground Aircraft!)")
            
    # ----------------------------------------------------------------
    # 5. COMPLETE FLEET DATA TRACKING TABLE
    # ----------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Complete Fleet Predictions Breakdown")
    
    # Clean up the dataframe columns to make it look professional for an airline manager
    ui_table = latest_states[['engine_id', 'cycle', 'Predicted_RUL']].copy()
    ui_table.columns = ['Engine ID Code', 'Last Logged Flight Cycle', 'Estimated Remaining Flights (RUL)']
    
    # Render the interactive data table
    st.dataframe(ui_table, use_container_width=True)

else:
    # Welcome banner displayed before a file is uploaded
    st.info("👋 Awaiting telemetry data stream. Please upload 'test_FD001.txt' in the left sidebar to initialize live tracking.")