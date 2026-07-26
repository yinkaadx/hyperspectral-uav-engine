import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time

st.set_page_config(page_title="Hyperspectral UAV Engine", layout="wide")

st.title("Serverless Hyperspectral Image Pipeline")
st.caption("Automated UAV Telemetry Ingestion, Spectral Unmixing & Land-Use Classification")

st.sidebar.header("UAV Mission Configuration")
selected_mission = st.sidebar.selectbox("Active Flight Operation", ["Agricultural Crop Health (NDVI Analysis)", "Mineral Exploration Survey", "Disaster Response (Flood Zone Mapping)"])
payload_size = st.sidebar.slider("Simulate Post-Flight Data Burst (Gigabytes)", 10, 100, 50)
run_simulation = st.sidebar.button("Initialize Automated GIS Pipeline")

st.sidebar.markdown("---")
st.sidebar.caption("Architecture: UAV Base Station -> AWS Lambda -> PCA Compression -> XGBoost")

if run_simulation:
    st.subheader(f"Active Remote Sensing Mission: {selected_mission}")
    
    col1, col2, col3, col4 = st.columns(4)
    metric_velocity = col1.empty()
    metric_bands = col2.empty()
    metric_compression = col3.empty()
    metric_status = col4.empty()

    chart_placeholder = st.empty()
    log_placeholder = st.empty()

    np.random.seed(2525)
    time_steps = pd.date_range(start=pd.Timestamp.now(), periods=100, freq="s")
    
    raw_data_volume = []
    classification_accuracy = []
    
    total_ingested = 0.0
    current_acc = 98.0
    
    for i in range(100):
        ingestion_rate = np.random.uniform(0.5, 1.5) * (payload_size / 20.0)
        total_ingested += ingestion_rate
        
        if i < 40:
            spectral_bands = int(np.random.uniform(200, 250))
            current_acc = 98.0 + np.random.uniform(-0.5, 0.5)
            status = "INGESTING & NORMALIZING"
        elif i >= 40 and i < 70:
            spectral_bands = int(np.random.uniform(15, 30)) 
            current_acc = 98.5 + np.random.uniform(-0.2, 0.2)
            status = "PCA DIMENSIONALITY REDUCTION"
        else:
            spectral_bands = int(np.random.uniform(15, 30))
            current_acc = 99.1 + np.random.uniform(-0.1, 0.1)
            status = "XGBOOST SPATIAL CLASSIFICATION"
            
        raw_data_volume.append(total_ingested)
        classification_accuracy.append(current_acc)
        
        compression_ratio = 100 - ((spectral_bands / 250.0) * 100)
        
        metric_velocity.metric("AWS Ingestion Volume (GB)", f"{total_ingested:.1f} GB")
        metric_bands.metric("Active Spectral Bands Processed", f"{spectral_bands} Bands")
        metric_compression.metric("In-Transit Data Compression", f"{compression_ratio:.1f}%", "Curse of Dimensionality Mitigated")
        
        if status == "XGBOOST SPATIAL CLASSIFICATION":
            metric_status.metric("Automated GIS Workflow", status, "Map Generation Complete")
        elif status == "PCA DIMENSIONALITY REDUCTION":
            metric_status.metric("Automated GIS Workflow", status, "Extracting Features")
        else:
            metric_status.metric("Automated GIS Workflow", status, "Reading Payload")
            
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_steps[:i+1], y=raw_data_volume, mode='lines', name='Data Ingested (GB)', fill='tozeroy', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=time_steps[:i+1], y=classification_accuracy, mode='lines', name='Model Accuracy (%)', yaxis='y2', line=dict(color='green', dash='dot')))
        
        fig.update_layout(
            title="Hyperspectral Data Pipeline: Serverless Burst Ingestion vs Classification Accuracy",
            xaxis=dict(title="High-Frequency Processing Timeline"),
            yaxis=dict(title="Data Volume Processed (GB)"),
            yaxis2=dict(title="Classification Accuracy (%)", overlaying='y', side='right', range=[95, 100]),
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
        
        if status == "PCA DIMENSIONALITY REDUCTION" and i == 40:
            log_placeholder.warning(f"COMPUTATIONAL SHIFT: Initiating serverless Principal Component Analysis at {time_steps[i].strftime('%H:%M:%S')}. Reducing 250 highly correlated spectral bands to 20 principal components to eliminate statistical noise.")
        elif status == "XGBOOST SPATIAL CLASSIFICATION" and i == 70:
            log_placeholder.success(f"CLASSIFICATION COMPLETE: Machine learning inference engine successfully mapped environmental signatures. Automatically exporting structured geodatabase to MAF Digital Lab web portal.")
        elif status == "INGESTING & NORMALIZING" and i % 5 == 0:
            log_placeholder.info(f"Log: Multi-gigabyte SPECIM camera payload {i} streaming via asynchronous AWS API. No desktop software required.")
            
        time.sleep(0.15)
        
    st.info("Simulation Complete. The cloud-native machine learning pipeline successfully automated the hyperspectral remote sensing workflow, eliminating manual GIS bottlenecks.")
else:
    st.info("Click 'Initialize Automated GIS Pipeline' in the sidebar to simulate high-velocity UAV data processing.")