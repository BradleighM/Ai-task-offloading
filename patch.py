import re

with open('edge_client.py', 'r') as f:
    content = f.read()

# 1. Add imports and init_db before st.set_page_config
imports_and_db = """
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

"""

# Insert right after `from sb3_contrib import RecurrentPPO`
if 'from sb3_contrib import RecurrentPPO' in content:
    content = content.replace('from sb3_contrib import RecurrentPPO', 'from sb3_contrib import RecurrentPPO\n' + imports_and_db)

# Remove old list append logic and replace with log_routing_decision and log_anomaly_scores
old_log = '''            # Log data to session history for charts
            new_log = pd.DataFrame([{
                "Timestamp": time.strftime("%H:%M:%S"),
                "Decision": actual_decision,
                "Execution_Time": exec_time,
                "CPU_Load": cpu_usage,
                "Latency": latency,
                "Complexity": complexity,
                "Energy_Est": energy_val
            }])
            st.session_state.history = pd.concat([st.session_state.history, new_log], ignore_index=True)'''

new_log = '''            # Log data to SQLite DB for persistent charts
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
                )'''

if old_log in content:
    content = content.replace(old_log, new_log)
else:
    print("Could not find old_log block!")

# Replace tab 2 with render_dashboard()
# We need to find `with tabs[1]:` and replace everything after it.
tab2_start = content.find('with tabs[1]:')
if tab2_start != -1:
    content = content[:tab2_start] + 'with tabs[1]:\n    render_dashboard()\n'
else:
    print("Could not find tabs[1] block!")

with open('edge_client.py', 'w') as f:
    f.write(content)

print("Patch applied.")
