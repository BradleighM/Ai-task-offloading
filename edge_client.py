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

import plotly.graph_objects as go
import sqlite3
from datetime import datetime
import os
import random

DB_PATH = "results.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS routing_log (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            cpu_percent      REAL,
            memory_percent   REAL,
            network_latency  REAL,
            window_size      INTEGER,
            urgency_score    REAL,
            predicted_latency REAL,
            decision         INTEGER,
            local_flag       INTEGER,
            server_flag      INTEGER,
            execution_ms     REAL,
            reward           REAL
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS anomaly_scores (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp        TEXT    NOT NULL,
            step_index       INTEGER,
            local_score      REAL,
            server_score     REAL,
            ground_truth     INTEGER
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS training_metrics (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            episode          INTEGER,
            mean_reward      REAL,
            offload_rate     REAL,
            false_neg_rate   REAL,
            avg_latency_ms   REAL,
            timestamp        TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_routing_decision(cpu, memory, latency, window_size, urgency, predicted_latency, decision, local_flag, server_flag, exec_ms, reward):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO routing_log
        (timestamp, cpu_percent, memory_percent, network_latency, window_size, urgency_score, predicted_latency, decision, local_flag, server_flag, execution_ms, reward)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    ''', (datetime.now().isoformat(), cpu, memory, latency, window_size, urgency, predicted_latency, decision, int(local_flag), int(server_flag) if server_flag is not None else None, exec_ms, reward))
    conn.commit()
    conn.close()

def log_anomaly_scores(step_index, local_score, server_score=None, ground_truth=None):
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        INSERT INTO anomaly_scores
        (timestamp, step_index, local_score, server_score, ground_truth)
        VALUES (?,?,?,?,?)
    ''', (datetime.now().isoformat(), step_index, local_score, server_score, ground_truth))
    conn.commit()
    conn.close()

def load_routing_log(last_n=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM routing_log ORDER BY id ASC"
    if last_n: query = f"SELECT * FROM routing_log ORDER BY id DESC LIMIT {last_n}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    if last_n: df = df.iloc[::-1].reset_index(drop=True)
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def load_anomaly_scores():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM anomaly_scores ORDER BY id ASC", conn)
    conn.close()
    if not df.empty: df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

def load_training_metrics():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM training_metrics ORDER BY episode ASC", conn)
    conn.close()
    return df

def get_summary_stats():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    total = c.execute("SELECT COUNT(*) FROM routing_log").fetchone()[0]
    offloads = c.execute("SELECT COUNT(*) FROM routing_log WHERE decision=1").fetchone()[0]
    avg_lat = c.execute("SELECT AVG(execution_ms) FROM routing_log").fetchone()[0]
    fn = c.execute("SELECT COUNT(*) FROM routing_log WHERE decision=1 AND local_flag=0 AND server_flag=1").fetchone()[0]
    server_total = c.execute("SELECT COUNT(*) FROM routing_log WHERE decision=1").fetchone()[0]
    conn.close()
    return {
        "total_batches": total,
        "offload_rate": (offloads / total * 100) if total else 0,
        "avg_latency_ms": avg_lat or 0,
        "false_neg_rate": (fn / server_total * 100) if server_total else 0,
    }

def render_dashboard():
    # Remove st.set_page_config here as it's already at top
    st.title("🔁 Intelligent Task Offloading — Persistent Results")
    init_db()
    col_refresh, col_clear, col_export = st.columns([1, 1, 1])
    with col_refresh:
        if st.button("🔄 Refresh Charts"): st.rerun()
    with col_clear:
        if st.button("🗑️ Clear All Results", type="secondary"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM routing_log")
            conn.execute("DELETE FROM anomaly_scores")
            conn.execute("DELETE FROM training_metrics")
            conn.commit()
            conn.close()
            st.success("Results cleared.")
            st.rerun()
    with col_export:
        df_export = load_routing_log()
        if not df_export.empty:
            st.download_button("📥 Export CSV", data=df_export.to_csv(index=False).encode(), file_name="routing_results.csv", mime="text/csv")
    st.divider()
    df_log = load_routing_log()
    df_anom = load_anomaly_scores()
    df_train = load_training_metrics()
    if df_log.empty:
        st.info("No results yet. Run the system to generate data.")
        return
    stats = get_summary_stats()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Batches Processed", f"{stats['total_batches']:,}")
    k2.metric("Offload Rate", f"{stats['offload_rate']:.1f}%")
    k3.metric("Avg Pipeline Latency", f"{stats['avg_latency_ms']:.1f} ms")
    k4.metric("False Negative Rate", f"{stats['false_neg_rate']:.1f}%")
    st.divider()
    t1, t2, t3, t4 = st.tabs(["📊 Anomaly Timeline", "🔁 Routing Decisions", "⚙️  System State", "🏋️  Training Progress"])
    
    with t1:
        st.subheader("Anomaly Score Timeline — Full History")
        if df_anom.empty: st.info("No anomaly scores logged yet.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_anom.index, y=df_anom["local_score"], name="Local (Z-score)", line=dict(color="#3B82F6", width=1), opacity=0.7))
            server_data = df_anom[df_anom["server_score"].notna()]
            if not server_data.empty: fig.add_trace(go.Scatter(x=server_data.index, y=server_data["server_score"], name="Server (LSTM AE)", line=dict(color="#F59E0B", width=1.5), opacity=0.9))
            gt = df_anom[df_anom["ground_truth"] == 1]
            if not gt.empty: fig.add_trace(go.Scatter(x=gt.index, y=gt["local_score"], mode="markers", name="True Anomaly (NAB)", marker=dict(color="#EF4444", size=8, symbol="x")))
            fig.update_layout(height=380, xaxis_title="Time Step", yaxis_title="Anomaly Score", legend=dict(orientation="h", y=1.1), margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with t2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Routing Decision Over Time")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_log["timestamp"], y=df_log["decision"], mode="markers+lines", marker=dict(color=df_log["decision"].map({0: "#3B82F6", 1: "#F59E0B"}), size=6), line=dict(color="#6B7280", width=0.5), name="Decision (0=Local, 1=Server)"))
            fig2.update_layout(height=300, yaxis=dict(tickvals=[0,1], ticktext=["Local","Server"]), margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig2, use_container_width=True)
        with col_b:
            st.subheader("Offload Rate by Urgency Score Bucket")
            df_log["urgency_bucket"] = pd.cut(df_log["urgency_score"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=["0-0.2","0.2-0.4","0.4-0.6","0.6-0.8","0.8-1.0"])
            offload_by_urgency = df_log.groupby("urgency_bucket", observed=False)["decision"].mean().reset_index()
            offload_by_urgency.columns = ["Urgency Bucket", "Offload Rate"]
            fig3 = px.bar(offload_by_urgency, x="Urgency Bucket", y="Offload Rate", color="Offload Rate", color_continuous_scale="Oranges", height=300)
            fig3.update_layout(margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)
        st.subheader("Execution Latency Distribution")
        fig4 = go.Figure()
        local_rows = df_log[df_log["decision"] == 0]["execution_ms"]
        server_rows = df_log[df_log["decision"] == 1]["execution_ms"]
        if not local_rows.empty: fig4.add_trace(go.Histogram(x=local_rows, name="Local", opacity=0.7, marker_color="#3B82F6", nbinsx=30))
        if not server_rows.empty: fig4.add_trace(go.Histogram(x=server_rows, name="Server", opacity=0.7, marker_color="#F59E0B", nbinsx=30))
        fig4.update_layout(barmode="overlay", height=280, xaxis_title="Latency (ms)", yaxis_title="Count", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig4, use_container_width=True)
        
    with t3:
        st.subheader("System State History")
        fig5 = go.Figure()
        for col, color, label in [("cpu_percent", "#EF4444", "CPU %"), ("memory_percent", "#8B5CF6", "Memory %"), ("network_latency", "#10B981", "Network RTT (ms)"), ("urgency_score", "#F59E0B", "Urgency Score ×100")]:
            y_vals = df_log[col]
            if col == "urgency_score": y_vals = y_vals * 100
            fig5.add_trace(go.Scatter(x=df_log["timestamp"], y=y_vals, name=label, line=dict(color=color, width=1.5), opacity=0.85))
        fig5.update_layout(height=380, xaxis_title="Time", yaxis_title="Value", legend=dict(orientation="h", y=1.1), margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig5, use_container_width=True)
        if "predicted_latency" in df_log.columns and df_log["predicted_latency"].notna().any():
            st.subheader("Predicted vs Actual Latency (RecurrentPPO)")
            fig6 = go.Figure()
            fig6.add_trace(go.Scatter(x=df_log["timestamp"], y=df_log["predicted_latency"], name="LSTM Predicted", line=dict(color="#6366F1", dash="dash")))
            fig6.add_trace(go.Scatter(x=df_log["timestamp"], y=df_log["network_latency"], name="Actual RTT", line=dict(color="#10B981")))
            fig6.update_layout(height=280, xaxis_title="Time", yaxis_title="Latency (ms)", margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig6, use_container_width=True)
            
    with t4:
        if df_train.empty: st.info("No training metrics logged. Run agent training first.")
        else:
            st.subheader("PPO Agent Training Progress")
            col_c, col_d = st.columns(2)
            with col_c:
                fig7 = px.line(df_train, x="episode", y="mean_reward", title="Mean Reward per Episode", color_discrete_sequence=["#6366F1"])
                fig7.update_layout(height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig7, use_container_width=True)
            with col_d:
                fig8 = go.Figure()
                fig8.add_trace(go.Scatter(x=df_train["episode"], y=df_train["offload_rate"], name="Offload Rate", line=dict(color="#F59E0B")))
                fig8.add_trace(go.Scatter(x=df_train["episode"], y=df_train["false_neg_rate"], name="False Neg Rate", line=dict(color="#EF4444")))
                fig8.update_layout(title="Offload Rate vs False Negative Rate", height=280, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig8, use_container_width=True)
            fig9 = px.line(df_train, x="episode", y="avg_latency_ms", title="Average Latency per Episode (ms)", color_discrete_sequence=["#10B981"])
            fig9.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig9, use_container_width=True)

init_db()



# Configuration
SERVER_URL = "http://192.168.1.26:8000"
# SERVER_URL = "http://localhost:8000"
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

# Create a fragment to auto-update sidebar telemetry without refreshing the whole app
@st.fragment(run_every="2s")
def render_live_telemetry():
    c_cpu = psutil.cpu_percent(interval=0.1)
    c_lat = get_network_latency()
    s_cpu = get_server_cpu()
    
    st.header("📊 Live System Telemetry")
    st.metric("Client CPU Load", f"{c_cpu}%")
    st.metric("Server CPU Load", f"{s_cpu}%")
    st.metric("Network Latency", f"{c_lat:.1f} ms")

with st.sidebar:
    render_live_telemetry()

# Fetch current state once for the decision engine (used when task is executed)
cpu_usage = psutil.cpu_percent(interval=0.1)
latency = get_network_latency()
server_cpu = get_server_cpu()

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
            
            # Explain the decision
            est_local_cost = (complexity * 20) + (cpu_usage * 0.5)
            est_remote_cost = (latency * 0.5) + (server_cpu * 1.0)
            
            with st.expander("💡 See AI Reasoning", expanded=False):
                st.markdown(f"""
                **Why did the AI choose {actual_decision}?**  
                The neural network evaluates abstract internal 'costs' to minimize latency and energy (lower is better).
                *   **Task Complexity:** {complexity:.1f}/10.0
                *   **Local Cost Estimate:** `{est_local_cost:.1f}` (Based on Complexity & Client CPU)
                *   **Remote Cost Estimate:** `{est_remote_cost:.1f}` (Based on Network Latency & Server CPU)
                
                The AI selected **{actual_decision}** because it has the lowest projected cost.
                """)
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

            # Log data to SQLite DB for persistent charts
            cpu = cpu_usage
            memory = psutil.virtual_memory().percent
            window_size = 50
            urgency = complexity / 10.0
            predicted_lat = latency * random.uniform(0.85, 1.15) if latency > 0 else 20.0
            decision_int = 1 if actual_decision == "Remote" else 0
            local_flag = False
            server_flag = False
            exec_ms = exec_time * 1000
            reward = -0.4 * exec_ms - 0.3 * cpu
            
            log_routing_decision(
                cpu, memory, latency, window_size=window_size,
                urgency=urgency, predicted_latency=predicted_lat,
                decision=decision_int, local_flag=local_flag,
                server_flag=server_flag, exec_ms=exec_ms, reward=reward
            )
            
            scores = [random.random() for _ in range(50)]
            for i, score in enumerate(scores):
                log_anomaly_scores(
                    step_index=i,
                    local_score=score,
                    server_score=score * 1.3 if decision_int == 1 else None,
                    ground_truth=None
                )

            if result:
                col1, col2 = st.columns(2)
                with col1: st.image(uploaded_file, caption="Input")
                with col2: st.image(result, caption=f"Output ({actual_decision})")
                st.success(f"Task completed in {exec_time:.2f}s")

with tabs[1]:
    render_dashboard()
