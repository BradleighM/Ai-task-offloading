import streamlit as st
import psutil
import requests
import cv2
import numpy as np
import time
import pandas as pd
import plotly.express as px
from PIL import Image
import io                                   
from stable_baselines3 import PPO                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          

# Configuration
SERVER_URL = "http://localhost:8000"
PROCESS_URL = f"{SERVER_URL}/process"

st.set_page_config(page_title="Intelligent AI Edge Analytics", layout="wide")

# --- 1. Data Logging (Persistent during the session) ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "Decision", "Execution_Time", "CPU_Load", "Latency", "Complexity", "Energy_Est"
    ])

# --- 2. Load the AI "Brain" ---
@st.cache_resource
def load_ai_model():
    try:
        # Loads the zip file created by train_agent.py
        return PPO.load("edge_ai_model")
    except Exception as e:
        return None

model = load_ai_model()

# --- 3. Real-Time Telemetry ---
def get_network_latency():
    """Measures actual round-trip time to the edge server."""
    try:
        start = time.time()
        requests.get(SERVER_URL, timeout=0.5)
        return (time.time() - start) * 1000 # Convert to ms
    except:
        return 500.0 # High penalty if server is unreachable

cpu_usage = psutil.cpu_percent(interval=0.1)
latency = get_network_latency()

# Sidebar Monitoring
st.sidebar.header("📊 Live System Telemetry")
st.sidebar.metric("Actual CPU Load", f"{cpu_usage}%")
st.sidebar.metric("Network Latency", f"{latency:.1f} ms")

# --- 4. Main UI with Tabs ---
st.title("🧠 Intelligent Edge System & Research Analytics")
tabs = st.tabs(["🚀 Real-Time Processor", "📈 Research Evaluation"])

with tabs[0]:
    st.header("Upload & Process")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    filter_choice = st.selectbox("Select Filter", ["Blur", "Edge Detection", "Grayscale"])

    if uploaded_file:
        img_pil = Image.open(uploaded_file)
        width, height = img_pil.size
        # Complexity = Megapixels (limited to 1-10 range)
        complexity = min(max((width * height) / 1_000_000, 1), 10)
        
        # ASK THE AI FOR THE DECISION
        if model is not None:
            obs = np.array([cpu_usage, latency, complexity], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            decision = "Remote" if action == 1 else "Local"
        else:
            decision = "Local (Model missing)"

        st.info(f"**AI Strategy Decision:** {decision}")

        if st.button("Execute Task", type="primary"):
            start_time = time.time()
            image_bytes = uploaded_file.getvalue()
            
            # --- Logic Execution ---
            if decision == "Remote":
                with st.spinner("Offloading to Edge Server..."):
                    files = {"file": ("img.jpg", image_bytes, "image/jpeg")}
                    try:
                        response = requests.post(PROCESS_URL, files=files, params={"filter_type": filter_choice})
                        result = response.content if response.status_code == 200 else None
                    except:
                        result = None
            else:
                with st.spinner("Processing on Local CPU..."):
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if filter_choice == "Blur":
                        processed = cv2.GaussianBlur(img, (51, 51), 0)
                    elif filter_choice == "Edge Detection":
                        processed = cv2.Canny(img, 100, 200)
                        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    else:
                        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    time.sleep(0.5) # Simulate local effort
                    _, encoded = cv2.imencode('.jpg', processed)
                    result = encoded.tobytes()

            exec_time = time.time() - start_time
            
            # --- Energy Modeling ---
            # Factor: Remote saves roughly 8x energy on the device side
            energy_factor = 0.1 if decision == "Remote" else 0.8
            energy_val = exec_time * energy_factor * (cpu_usage/100 + 1)

            # Log data to session history for charts
            new_log = pd.DataFrame([{
                "Timestamp": time.strftime("%H:%M:%S"),
                "Decision": decision,
                "Execution_Time": exec_time,
                "CPU_Load": cpu_usage,
                "Latency": latency,
                "Complexity": complexity,
                "Energy_Est": energy_val
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_log], ignore_index=True)

            if result:
                col1, col2 = st.columns(2)
                with col1: st.image(uploaded_file, caption="Input")
                with col2: st.image(result, caption=f"Output ({decision})")
                st.success(f"Task completed in {exec_time:.2f}s")

with tabs[1]:
    st.header("📊 Performance & Sustainability Metrics")
    
    if not st.session_state.history.empty:
        df = st.session_state.history
        
        # Row 1: Decisions and Time
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Decision Distribution")
            fig_pie = px.pie(df, names='Decision', color='Decision', 
                             color_discrete_map={'Local':'#EF553B', 'Remote':'#00CC96'})
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("Execution Speed (Seconds)")
            fig_line = px.line(df, y="Execution_Time", markers=True, labels={"index": "Task #", "Execution_Time": "Seconds"})
            st.plotly_chart(fig_line, use_container_width=True)

        # Row 2: Sustainability Model
        st.subheader("⚡ Cumulative Energy Saving Analysis")
        # Comparative model: AI vs a hypothetical "Always Local" approach
        df['Baseline_Energy'] = df['Execution_Time'] * 0.8 * (df['CPU_Load']/100 + 1)
        
        comparison = pd.DataFrame({
            "Approach": ["AI-Optimized Engine", "Traditional (Always Local)"],
            "Total Energy (Units)": [df['Energy_Est'].sum(), df['Baseline_Energy'].sum()]
        })
        
        fig_bar = px.bar(comparison, x='Approach', y='Total Energy (Units)', color='Approach',
                        color_discrete_map={"AI-Optimized Engine": "#00CC96", "Traditional (Always Local)": "#EF553B"})
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.write("### Experiment Data Logs")
        st.dataframe(df)
    else:
        st.warning("No data collected yet. Please upload and process images in the first tab.")