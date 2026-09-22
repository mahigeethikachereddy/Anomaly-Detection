# 🏭 Predictive Maintenance Dashboard — Anomaly Detection

A machine learning-powered predictive maintenance system that detects anomalies and predicts equipment failures using sensor data from manufacturing machines.

## 📋 Overview

This project uses AI to predict machine failures before they happen. It combines multiple anomaly detection algorithms and a trained neural network model to provide real-time failure probability predictions from sensor readings.

### Key Features
- 🎯 **Real-Time Prediction** — Input sensor readings and get instant failure probability
- 🔄 **Live Simulation** — Auto-updating simulated sensor data stream with trend charts
- 📈 **Data Analytics** — Explore historical data with interactive visualizations
- 📊 **Model Performance** — View accuracy, ROC-AUC, confusion matrix, and feature importance
- 📋 **Batch Prediction** — Upload CSV files for bulk predictions with risk classification

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Core language |
| Streamlit | Interactive web dashboard |
| Scikit-learn | ML models (Random Forest, Isolation Forest) |
| PyTorch | Neural network model |
| Pandas / NumPy | Data processing |
| Plotly | Interactive visualizations |
| Joblib | Model serialization |

## 📊 Dataset

The **ai4i2020.csv** dataset contains sensor readings from manufacturing machines:
- **Features**: Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min], Machine Type (L/M/H)
- **Target**: Machine Failure (binary) + individual failure types (TWF, HDF, PWF, OSF, RNF)
- **Records**: ~10,000+ samples across 3 machine types

## 📁 Project Structure

```
├── app.py                          # Main Streamlit dashboard
├── ai4i2020.csv                    # Manufacturing sensor dataset
├── predictive_maintenance_model.pkl # Trained Random Forest model
├── isolation_forest_model.pkl      # Anomaly detection model
├── neural_network_model.pth        # PyTorch neural network
├── feature_scaler.pkl              # Feature preprocessing scaler
├── anomaly_scaler.pkl              # Anomaly detection scaler
├── label_encoder.pkl               # Machine type label encoder
├── correlation_heatmap.png         # Feature correlation visualization
├── data_distribution.png           # Sensor data distributions
├── feature_importance.png          # Model feature importance chart
├── training_history.png            # Model training curves
└── .qodo/                          # Configuration directory
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
pip install streamlit pandas numpy plotly scikit-learn torch joblib
```

### Run the Dashboard
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

## 📈 Model Performance

| Model | Accuracy | ROC-AUC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| Random Forest | 97.87% | 0.9747 | 86% | 46% |
| Neural Network | 98.10% | 0.9782 | 83% | 54% |

### Top Features
1. **Torque** — 30.6% importance
2. **Rotational Speed** — 29.6% importance
3. **Tool Wear** — 19.7% importance
4. **Air Temperature** — 11.4% importance
5. **Process Temperature** — 7.0% importance
6. **Machine Type** — 1.6% importance

## 🤖 Anomaly Detection

The project implements two complementary approaches:
1. **Isolation Forest** — Unsupervised anomaly detection for identifying unusual sensor patterns
2. **Supervised ML Models** — Random Forest and Neural Network for predicting specific failure types

## 🔧 Usage

### Real-Time Prediction
Enter sensor values (temperature, speed, torque, tool wear, machine type) to get a failure probability score with risk classification (Low/Medium/High).

### Live Simulation
Enable auto-refresh to simulate streaming sensor data with real-time trend visualization and risk alerts.

### Batch Prediction
Upload a CSV with sensor readings to predict failure probabilities for multiple machines at once.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Mahi Geethika Kachereddy**
- GitHub: [@mahigeethikachereddy](https://github.com/mahigeethikachereddy)
- Repository: [Anomaly-Detection](https://github.com/mahigeethikachereddy/Anomaly-Detection)
