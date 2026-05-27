"""
cpu_stress.py  —  Demo CPU Stress Tool
--------------------------------------
Hammers the CPU using multiple threads so the PPO agent is forced to offload.

Usage:
    python cpu_stress.py          # stress at full power (default 4 threads)
    python cpu_stress.py --threads 2    # lighter stress
    python cpu_stress.py --stop         # (just re-run the script; Ctrl+C to stop)
"""

import threading
import time
import sys
import psutil

# ── Config ────────────────────────────────────────────────────────────────────
THREADS   = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
TARGET_PC = 90          # rough target CPU % (informational only)
DURATION  = 120         # auto-stop after this many seconds (0 = run forever)
# ─────────────────────────────────────────────────────────────────────────────

stop_event = threading.Event()


def burn_cpu():
    """Tight arithmetic loop — pegs one CPU core."""
    x = 0.0
    while not stop_event.is_set():
        x = (x * 1.0000001 + 3.14159) ** 0.5   # just enough to avoid optimisation


def monitor():
    """Print live CPU % every second so you can see it climbing."""
    print("\n🔥  CPU STRESS ACTIVE — press Ctrl+C to stop\n")
    print(f"{'Time':>6}  {'CPU %':>6}  {'RAM %':>6}")
    print("-" * 26)
    elapsed = 0
    while not stop_event.is_set():
        cpu = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory().percent
        bar = "█" * int(cpu / 5)
        print(f"{elapsed:>5}s  {cpu:>5.1f}%  {ram:>5.1f}%  {bar}")
        elapsed += 1
        if DURATION > 0 and elapsed >= DURATION:
            print(f"\n⏱  Auto-stopping after {DURATION}s.")
            stop_event.set()


# ── Launch ────────────────────────────────────────────────────────────────────
threads = [threading.Thread(target=burn_cpu, daemon=True) for _ in range(THREADS)]
monitor_thread = threading.Thread(target=monitor, daemon=True)

try:
    for t in threads:
        t.start()
    monitor_thread.start()
    monitor_thread.join()
except KeyboardInterrupt:
    print("\n\n✅  Stress stopped.  CPU will drop back to normal.")
    stop_event.set()