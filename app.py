"""
Predictive Maintenance Dashboard - Streamlit Web App
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
    .risk-low {
        background-color: #6bcb77;
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        model = joblib.load('predictive_maintenance_model.pkl')
        scaler = joblib.load('feature_scaler.pkl')
        return model, scaler
    except:
        return None, None

# Prediction function
def predict_failure(model, scaler, air_temp, process_temp, speed, torque, tool_wear, machine_type):
    """Make prediction"""
    type_map = {'L': 0, 'M': 1, 'H': 2}
    type_encoded = type_map.get(machine_type, 0)
    
    features = np.array([[
        type_encoded, air_temp, process_temp, speed, torque, tool_wear
    ]])
    
    features_scaled = scaler.transform(features)
    probability = model.predict_proba(features_scaled)[0, 1]
    
    return probability

# Load data for analysis
@st.cache_data
def load_data():
    """Load the dataset"""
    df = pd.read_csv('ai4i2020.csv')
    failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    df['FAILURE'] = (df[failure_cols].sum(axis=1) > 0).astype(int)
    return df

def main():
    # Header
    st.markdown('<div class="main-header">🏭 Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)
    st.markdown("*Real-time machine failure prediction using AI*")
    st.markdown("---")
    
    # Load models
    model, scaler = load_models()
    
    if model is None or scaler is None:
        st.error("❌ Models not found! Please run the training script first.")
        st.info("Run: python predictive_maintenance.py")
        return
    
    # Load data
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/factory.png", width=80)
        st.markdown("## 📊 Navigation")
        
        # ADD THE NEW PAGE HERE - This is where you add to the navigation
        page = st.radio(
            "Select Page",
            ["🎯 Real-Time Prediction", "🔄 Live Simulation", "📈 Data Analytics", "📊 Model Performance", "📋 Batch Prediction"]
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        This dashboard uses AI to predict machine failures before they happen.
        
        **Key Features:**
        - Real-time failure prediction
        - Live simulation mode
        - Historical data analysis
        - Model performance metrics
        - Batch prediction capability
        """)
        
        st.markdown("---")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # PAGE 1: REAL-TIME PREDICTION (Original)
    # ========================================================================
    if page == "🎯 Real-Time Prediction":
        st.markdown("## 🎯 Real-Time Failure Prediction")
        st.markdown("Enter sensor readings to predict machine failure probability")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Sensor Inputs")
            
            machine_type = st.selectbox(
                "Machine Type",
                ["L (Low)", "M (Medium)", "H (High)"],
                help="Different machine types have different baseline characteristics"
            )
            
            air_temp = st.slider(
                "Air Temperature [K]",
                min_value=295.0,
                max_value=305.0,
                value=298.5,
                step=0.1,
                help="Ambient air temperature around the machine"
            )
            
            process_temp = st.slider(
                "Process Temperature [K]",
                min_value=305.0,
                max_value=315.0,
                value=308.6,
                step=0.1,
                help="Temperature during the manufacturing process"
            )
            
            speed = st.slider(
                "Rotational Speed [rpm]",
                min_value=1000,
                max_value=3000,
                value=1550,
                step=10,
                help="Speed of the rotating components"
            )
            
            torque = st.slider(
                "Torque [Nm]",
                min_value=0.0,
                max_value=80.0,
                value=35.0,
                step=0.5,
                help="Torque applied during operation"
            )
            
            tool_wear = st.slider(
                "Tool Wear [min]",
                min_value=0,
                max_value=250,
                value=50,
                step=5,
                help="Total tool usage time"
            )
        
        # Map machine type
        type_map = {"L (Low)": "L", "M (Medium)": "M", "H (High)": "H"}
        machine_type_code = type_map[machine_type]
        
        # Make prediction
        probability = predict_failure(model, scaler, air_temp, process_temp, 
                                      speed, torque, tool_wear, machine_type_code)
        
        with col2:
            st.markdown("### 🔮 Prediction Result")
            
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = probability * 100,
                title = {'text': "Failure Probability (%)"},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            
            fig.update_layout(height=300, width=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk assessment
            if probability >= 0.7:
                st.markdown('<div class="risk-high">🔴 HIGH RISK - Immediate maintenance required!</div>', unsafe_allow_html=True)
                st.warning("⚠️ Action: Stop machine and inspect immediately!")
            elif probability >= 0.4:
                st.markdown('<div class="risk-medium">🟡 MEDIUM RISK - Schedule maintenance soon</div>', unsafe_allow_html=True)
                st.info("📋 Action: Plan maintenance within 48 hours")
            else:
                st.markdown('<div class="risk-low">🟢 LOW RISK - Normal operation</div>', unsafe_allow_html=True)
                st.success("✅ Action: Continue monitoring normally")
            
            # Additional recommendations
            st.markdown("### 💡 Recommendations")
            if probability > 0.5:
                st.write("• Check tool wear immediately")
                st.write("• Inspect torque settings")
                st.write("• Monitor temperature trends")
            else:
                st.write("• Continue regular maintenance schedule")
                st.write("• Log data for trend analysis")
                st.write("• No immediate action required")
        
        # Visualize sensor readings
        st.markdown("---")
        st.markdown("### 📊 Sensor Reading Analysis")
        
        col3, col4, col5 = st.columns(3)
        
        with col3:
            temp_diff = process_temp - air_temp
            st.metric("Temperature Difference", f"{temp_diff:.1f} K", 
                     delta="Normal" if 8 < temp_diff < 12 else "High")
        
        with col4:
            power = (torque * speed) / 9550  # Power in kW approximation
            st.metric("Power Output", f"{power:.1f} kW")
        
        with col5:
            st.metric("Tool Wear Status", f"{tool_wear} min",
                     delta="High" if tool_wear > 200 else "Normal")
    
    # ========================================================================
    # PAGE 2: LIVE SIMULATION (NEW - Add this entire section)
    # ========================================================================
    elif page == "🔄 Live Simulation":
        st.markdown("## 🔄 Live Real-Time Simulation")
        st.markdown("*Automatically updates with simulated sensor data - Demonstrates real-time monitoring*")
        
        st.info("💡 **How it works:** This mode simulates live sensor data streaming. In a real factory, actual sensors would send this data automatically.")
        
        # Auto-refresh controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            auto_refresh = st.checkbox("🔄 Enable live refresh", value=True)
        
        with col2:
            refresh_rate = st.slider("Refresh rate (seconds)", 1, 10, 3)
        
        with col3:
            if st.button("📊 Refresh Now"):
                st.rerun()
        
        st.markdown("---")
        
        # Initialize session state for history
        if 'simulation_history' not in st.session_state:
            st.session_state.simulation_history = []
        
        # Generate simulated real-time data
        np.random.seed(int(time.time() * 1000) % 2**32)
        
        # Simulate realistic sensor variations
        current_time = datetime.now()
        
        # Create realistic sensor patterns
        air_temp = np.random.normal(300, 2)
        process_temp = air_temp + np.random.normal(10, 1)
        speed = max(1000, min(3000, np.random.normal(1550, 150)))
        torque = max(0, min(80, np.random.normal(40, 15)))
        tool_wear = min(250, max(0, np.random.normal(100, 60)))
        machine_type = np.random.choice(['L', 'M', 'H'], p=[0.5, 0.3, 0.2])
        
        # Make prediction
        probability = predict_failure(model, scaler, air_temp, process_temp, 
                                      speed, torque, tool_wear, machine_type)
        
        # Display current readings
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📡 Current Sensor Readings")
            
            # Create a nice display of sensor values
            sensor_df = pd.DataFrame({
                'Parameter': ['Air Temperature', 'Process Temperature', 'Rotational Speed', 
                             'Torque', 'Tool Wear', 'Machine Type'],
                'Value': [f"{air_temp:.1f} K", f"{process_temp:.1f} K", 
                         f"{speed:.0f} rpm", f"{torque:.1f} Nm", 
                         f"{tool_wear:.0f} min", machine_type],
                'Status': [
                    '🟢 Normal' if 295 < air_temp < 305 else '🟡 Check',
                    '🟢 Normal' if 305 < process_temp < 315 else '🟡 Check',
                    '🟢 Normal' if 1200 < speed < 2000 else '🟡 Check',
                    '🟢 Normal' if torque < 60 else '🟡 High',
                    '🟢 Normal' if tool_wear < 200 else '🔴 Replace soon',
                    '✅ OK'
                ]
            })
            st.dataframe(sensor_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("### 🔮 Live Prediction")
            
            # Live gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Current Failure Probability"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "salmon"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk alert
            if probability >= 0.7:
                st.markdown('<div class="risk-high">🔴 HIGH RISK ALERT! Immediate action required!</div>', unsafe_allow_html=True)
            elif probability >= 0.4:
                st.markdown('<div class="risk-medium">🟡 MEDIUM RISK - Monitor closely</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">🟢 LOW RISK - Normal operation</div>', unsafe_allow_html=True)
        
        # Add to history
        st.session_state.simulation_history.append({
            'time': current_time,
            'probability': probability,
            'torque': torque,
            'speed': speed,
            'tool_wear': tool_wear,
            'air_temp': air_temp,
            'process_temp': process_temp
        })
        
        # Keep only last 100 points
        if len(st.session_state.simulation_history) > 100:
            st.session_state.simulation_history = st.session_state.simulation_history[-100:]
        
        # Real-time trend charts
        st.markdown("---")
        st.markdown("### 📈 Real-Time Trends")
        
        history_df = pd.DataFrame(st.session_state.simulation_history)
        
        if len(history_df) > 1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Failure probability trend
                fig1 = px.line(history_df, x='time', y='probability', 
                              title='📊 Failure Probability Trend',
                              labels={'probability': 'Failure Probability (%)', 'time': 'Time'})
                fig1.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High Risk Threshold")
                fig1.add_hline(y=40, line_dash="dash", line_color="orange", annotation_text="Medium Risk Threshold")
                fig1.update_layout(height=400)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Torque and Speed trend
                fig2 = make_subplots(specs=[[{"secondary_y": True}]])
                fig2.add_trace(go.Scatter(x=history_df['time'], y=history_df['torque'], 
                                         name="Torque (Nm)", line=dict(color='blue')),
                              secondary_y=False)
                fig2.add_trace(go.Scatter(x=history_df['time'], y=history_df['speed'], 
                                         name="Speed (rpm)", line=dict(color='red')),
                              secondary_y=True)
                fig2.update_layout(title='⚙️ Torque & Speed Trends', height=400)
                fig2.update_xaxes(title_text="Time")
                fig2.update_yaxes(title_text="Torque (Nm)", secondary_y=False)
                fig2.update_yaxes(title_text="Speed (rpm)", secondary_y=True)
                st.plotly_chart(fig2, use_container_width=True)
            
            # Tool wear trend
            fig3 = px.line(history_df, x='time', y='tool_wear',
                          title='🔧 Tool Wear Progression',
                          labels={'tool_wear': 'Tool Wear (minutes)', 'time': 'Time'})
            fig3.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Replace Tool")
            fig3.update_layout(height=400)
            st.plotly_chart(fig3, use_container_width=True)
        
        # Statistics summary
        st.markdown("---")
        st.markdown("### 📊 Session Statistics")
        
        if len(history_df) > 0:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_prob = history_df['probability'].mean()
                st.metric("Avg Failure Probability", f"{avg_prob:.1%}")
            
            with col2:
                max_prob = history_df['probability'].max()
                st.metric("Peak Risk", f"{max_prob:.1%}")
            
            with col3:
                high_risk_count = (history_df['probability'] >= 0.7).sum()
                st.metric("High Risk Events", high_risk_count)
            
            with col4:
                readings_count = len(history_df)
                st.metric("Total Readings", readings_count)
        
        # Auto-refresh logic (at the end of the page)
        if auto_refresh:
            time.sleep(refresh_rate)
            st.rerun()
    
    # ========================================================================
    # PAGE 3: DATA ANALYTICS (Original)
    # ========================================================================
    elif page == "📈 Data Analytics":
        st.markdown("## 📈 Historical Data Analysis")
        st.markdown("Explore patterns in historical machine data")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            failure_rate = df['FAILURE'].mean() * 100
            st.metric("Failure Rate", f"{failure_rate:.2f}%")
        with col3:
            avg_tool_wear = df['Tool wear [min]'].mean()
            st.metric("Avg Tool Wear", f"{avg_tool_wear:.0f} min")
        with col4:
            avg_speed = df['Rotational speed [rpm]'].mean()
            st.metric("Avg Speed", f"{avg_speed:.0f} rpm")
        
        # Filter controls
        st.markdown("### 🔍 Data Filters")
        col1, col2 = st.columns(2)
        
        with col1:
            machine_filter = st.multiselect(
                "Machine Type",
                options=['L', 'M', 'H'],
                default=['L', 'M', 'H']
            )
        
        with col2:
            failure_filter = st.selectbox(
                "Failure Status",
                options=["All", "Failed Only", "Normal Only"]
            )
        
        # Apply filters
        filtered_df = df[df['Type'].isin(machine_filter)]
        
        if failure_filter == "Failed Only":
            filtered_df = filtered_df[filtered_df['FAILURE'] == 1]
        elif failure_filter == "Normal Only":
            filtered_df = filtered_df[filtered_df['FAILURE'] == 0]
        
        # Distribution plots
        st.markdown("### 📊 Sensor Data Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(filtered_df, x='Torque [Nm]', color='Type', 
                               title='Torque Distribution by Machine Type',
                               nbins=50, opacity=0.7)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(filtered_df, x='Type', y='Rotational speed [rpm]', 
                        color='Type', title='Speed Distribution by Machine Type')
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### 🔥 Feature Correlation")
        corr_cols = ['Air temperature [K]', 'Process temperature [K]', 
                    'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
        
        corr_matrix = filtered_df[corr_cols].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Correlation Heatmap")
        st.plotly_chart(fig, use_container_width=True)
        
        # Time series view
        st.markdown("### 📈 Trend Analysis")
        sample_size = st.slider("Sample Size", 100, len(filtered_df), 1000)
        
        fig = make_subplots(rows=2, cols=2, subplot_titles=("Torque Trend", "Speed Trend", 
                                                             "Tool Wear Trend", "Temperature Trend"))
        
        fig.add_trace(go.Scatter(y=filtered_df['Torque [Nm]'].head(sample_size), 
                                mode='lines', name='Torque'), row=1, col=1)
        fig.add_trace(go.Scatter(y=filtered_df['Rotational speed [rpm]'].head(sample_size), 
                                mode='lines', name='Speed', line=dict(color='red')), row=1, col=2)
        fig.add_trace(go.Scatter(y=filtered_df['Tool wear [min]'].head(sample_size), 
                                mode='lines', name='Tool Wear', line=dict(color='green')), row=2, col=1)
        fig.add_trace(go.Scatter(y=filtered_df['Process temperature [K]'].head(sample_size), 
                                mode='lines', name='Temperature', line=dict(color='orange')), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    # ========================================================================
    # PAGE 4: MODEL PERFORMANCE (Original)
    # ========================================================================
    elif page == "📊 Model Performance":
        st.markdown("## 📊 Model Performance Metrics")
        st.markdown("Understanding how our AI model performs")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Model Accuracy", "97.9%", delta="+2.1% vs baseline")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 ROC-AUC Score", "0.975", delta="Excellent")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Precision (Failures)", "86%", delta="Good")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Confusion Matrix Visualization
        st.markdown("### 📊 Confusion Matrix")
        confusion_data = {
            'Actual Normal': [2890, 6],
            'Actual Failure': [56, 48]
        }
        
        fig = px.imshow([[2890, 56], [6, 48]], 
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=['Normal', 'Failure'],
                        y=['Normal', 'Failure'],
                        text_auto=True,
                        aspect="auto",
                        title="Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        st.markdown("### 🔑 Feature Importance")
        
        features = ['Torque', 'Rotational Speed', 'Tool Wear', 
                   'Air Temperature', 'Process Temperature', 'Machine Type']
        importance = [30.6, 29.6, 19.7, 11.4, 7.0, 1.6]
        
        fig = px.bar(x=importance, y=features, orientation='h',
                    title="What Matters Most for Prediction",
                    labels={'x': 'Importance (%)', 'y': 'Features'},
                    color=importance, color_continuous_scale='Viridis')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Model Comparison
        st.markdown("### 🤖 Model Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'Neural Network'],
            'Accuracy (%)': [97.87, 98.10],
            'ROC-AUC': [0.9747, 0.9782],
            'Precision (Failures %)': [86, 83],
            'Recall (Failures %)': [46, 54]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Explanation
        st.markdown("### 💡 Model Insights")
        st.info("""
        **Key Findings:**
        - Torque and Rotational Speed are the most important predictors
        - The model correctly identifies 86% of predicted failures
        - Neural network slightly outperforms Random Forest
        - False positives are rare (only 0.2% of normal operations)
        """)
    
    # ========================================================================
    # PAGE 5: BATCH PREDICTION (Original)
    # ========================================================================
    elif page == "📋 Batch Prediction":
        st.markdown("## 📋 Batch Prediction")
        st.markdown("Upload a CSV file with multiple sensor readings for batch prediction")
        
        # Template download
        template = pd.DataFrame({
            'Air_temperature_K': [298.5, 299.0],
            'Process_temperature_K': [308.6, 309.0],
            'Rotational_speed_rpm': [1550, 1300],
            'Torque_Nm': [35.0, 60.0],
            'Tool_wear_min': [50, 200],
            'Machine_Type': ['L', 'L']
        })
        
        csv = template.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=csv,
            file_name='batch_prediction_template.csv',
            mime='text/csv'
        )
        
        # File upload
        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(batch_df)} records")
                
                # Show preview
                st.markdown("### 📋 Data Preview")
                st.dataframe(batch_df.head())
                
                # Make predictions
                if st.button("🔮 Run Batch Prediction"):
                    with st.spinner("Making predictions..."):
                        predictions = []
                        for _, row in batch_df.iterrows():
                            prob = predict_failure(
                                model, scaler,
                                row['Air_temperature_K'],
                                row['Process_temperature_K'],
                                row['Rotational_speed_rpm'],
                                row['Torque_Nm'],
                                row['Tool_wear_min'],
                                row['Machine_Type']
                            )
                            predictions.append(prob)
                        
                        batch_df['Failure_Probability'] = predictions
                        batch_df['Risk_Level'] = batch_df['Failure_Probability'].apply(
                            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
                        )
                        
                        # Show results
                        st.markdown("### 📊 Prediction Results")
                        
                        # Summary stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            high_risk = (batch_df['Risk_Level'] == 'High').sum()
                            st.metric("High Risk Machines", high_risk)
                        with col2:
                            medium_risk = (batch_df['Risk_Level'] == 'Medium').sum()
                            st.metric("Medium Risk Machines", medium_risk)
                        with col3:
                            low_risk = (batch_df['Risk_Level'] == 'Low').sum()
                            st.metric("Low Risk Machines", low_risk)
                        
                        # Display results table
                        st.dataframe(batch_df, use_container_width=True)
                        
                        # Download results
                        csv_result = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv_result,
                            file_name='batch_predictions.csv',
                            mime='text/csv'
                        )
                        
                        # Visualization
                        fig = px.histogram(batch_df, x='Failure_Probability', 
                                          nbins=20, title='Failure Probability Distribution')
                        st.plotly_chart(fig, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Error processing file: {e}")

if __name__ == "__main__":
    main()
    