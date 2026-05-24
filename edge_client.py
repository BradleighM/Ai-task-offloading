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
from sb3_contrib import RecurrentPPO

# Configuration
# SERVER_URL = "http://172.20.10.3:8000
SERVER_URL = "http://localhost:8000"
PROCESS_URL = f"{SERVER_URL}/process"

st.set_page_config(page_title="Intelligent AI Edge Analytics", layout="wide", page_icon="⚡")

# --- Custom Premium UI Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&display=swap');

    /* Global Typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main App Background (Dark Gradient) */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }

    /* Headers */
    h1 {
        background: -webkit-linear-gradient(45deg, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #cbd5e1;
        font-weight: 600;
    }

    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Metrics Styling */
    [data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Primary Execute Button */
    button[kind="primary"] {
        background: linear-gradient(45deg, #4f46e5, #ec4899) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.5rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(236, 72, 153, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(236, 72, 153, 0.6) !important;
    }

    /* Custom Tabs */
    [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: transparent;
    }
    [data-baseweb="tab"] {
        background-color: transparent !important;
        border-radius: 8px !important;
        color: #94a3b8 !important;
        border: 1px solid transparent !important;
        transition: all 0.2s;
    }
    [data-baseweb="tab"]:hover {
        color: #f8fafc !important;
        background-color: rgba(255,255,255,0.05) !important;
    }
    [aria-selected="true"] {
        color: #fff !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        background: rgba(255,255,255,0.1) !important;
        backdrop-filter: blur(4px);
    }

    /* Status Alerts & Expanders */
    .stAlert {
        border-radius: 12px !important;
        border: none !important;
        background: rgba(255, 255, 255, 0.05) !important;
        backdrop-filter: blur(10px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
        color: #f8fafc !important;
    }
    
    /* File Uploader area */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255,255,255,0.03) !important;
        border: 1px dashed rgba(255,255,255,0.2) !important;
        border-radius: 12px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: #38bdf8 !important;
        background: rgba(255,255,255,0.08) !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 1. Data Logging (Persistent during the session) ---
if 'history' not in st.session_state:
    st.session_state.history = pd.DataFrame(columns=[
        "Timestamp", "Decision", "Execution_Time", "CPU_Load", "Latency", "Complexity", "Energy_Est"
    ])

if 'lstm_states' not in st.session_state:
    st.session_state.lstm_states = None
if 'episode_starts' not in st.session_state:
    st.session_state.episode_starts = np.ones((1,), dtype=bool)

# --- 2. Load the AI "Brain" ---
@st.cache_resource
def load_ai_model():
    try:
        return RecurrentPPO.load("edge_ai_model")
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

def get_server_cpu():
    """Fetches the actual CPU load from the edge server."""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=0.5)
        if response.status_code == 200:
            return response.json().get("cpu_load", 0.0)
    except:
        pass
    return 100.0 # Heavy penalty if server is unreachable

cpu_usage = psutil.cpu_percent(interval=0.1)
latency = get_network_latency()
server_cpu = get_server_cpu()

# Sidebar Monitoring
st.sidebar.header("📊 Live System Telemetry")
st.sidebar.metric("Client CPU Load", f"{cpu_usage}%")
st.sidebar.metric("Server CPU Load", f"{server_cpu}%")
st.sidebar.metric("Network Latency", f"{latency:.1f} ms")

# --- NEW: Presentation Controls (The Toggle Switch) ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Testing Controls")
override_ai = st.sidebar.checkbox(
    "Override AI (Force Local Only)", 
    value=False, 
    help="Check this to simulate a 'dumb' app that refuses to offload."
)

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
        
        # --- DECISION ENGINE LOGIC ---
        if override_ai:
            actual_decision = "Local"
            st.error("⚠️ **AI Disabled:** Forcing Local Execution (Simulating 'Dumb' App)")
        elif model is not None:
            obs = np.array([cpu_usage, latency, complexity, server_cpu], dtype=np.float32)
            # Use recurrent state for predictive routing
            action, st.session_state.lstm_states = model.predict(
                obs, 
                state=st.session_state.lstm_states, 
                episode_start=st.session_state.episode_starts,
                deterministic=True
            )
            # Reset episode start flag after first prediction
            st.session_state.episode_starts = np.zeros((1,), dtype=bool)
            
            actual_decision = "Remote" if action == 1 else "Local"
            st.info(f"**AI Strategy Decision:** {actual_decision} 🧠 *(Powered by Predictive LSTM Memory)*")
        else:
            actual_decision = "Local"
            st.warning("Model missing, defaulting to Local.")

        if st.button("Execute Task", type="primary"):
            start_time = time.time()
            image_bytes = uploaded_file.getvalue()
            
            # --- Logic Execution ---
            if actual_decision == "Remote":
                with st.spinner("🌐 Offloading to Edge Server..."):
                    files = {"file": ("img.jpg", image_bytes, "image/jpeg")}
                    try:
                        response = requests.post(PROCESS_URL, files=files, params={"filter_type": filter_choice})
                        result = response.content if response.status_code == 200 else None
                    except:
                        result = None
            else:
                with st.spinner("💻 Processing on Local CPU..."):
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if filter_choice == "Blur":
                        processed = cv2.GaussianBlur(img, (51, 51), 0)
                    elif filter_choice == "Edge Detection":
                        processed = cv2.Canny(img, 100, 200)
                        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                    else:
                        processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    time.sleep(0.5) # Simulate heavy local effort
                    _, encoded = cv2.imencode('.jpg', processed)
                    result = encoded.tobytes()

            exec_time = time.time() - start_time
            
            # --- Energy Modeling ---
            # Factor: Remote saves roughly 8x energy on the device side
            energy_factor = 0.1 if actual_decision == "Remote" else 0.8
            energy_val = exec_time * energy_factor * (cpu_usage/100 + 1)

            # Log data to session history for charts
            new_log = pd.DataFrame([{
                "Timestamp": time.strftime("%H:%M:%S"),
                "Decision": actual_decision,
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
                with col2: st.image(result, caption=f"Output ({actual_decision})")
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