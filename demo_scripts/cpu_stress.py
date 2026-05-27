"""
cpu_stress.py  —  Demo CPU Stress Tool
--------------------------------------
Hammers the CPU using multiple processes so the PPO agent is forced to offload.

Usage:
    python cpu_stress.py          # stress at full power (default 4 processes)
    python cpu_stress.py --threads 2    # lighter stress
    python cpu_stress.py --stop         # (just re-run the script; Ctrl+C to stop)
"""

import multiprocessing
import time
import sys
import psutil

# ── Config ────────────────────────────────────────────────────────────────────
THREADS   = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 4
TARGET_PC = 90          # rough target CPU % (informational only)
DURATION  = 120         # auto-stop after this many seconds (0 = run forever)
# ─────────────────────────────────────────────────────────────────────────────

def burn_cpu():
    """Tight arithmetic loop — pegs one CPU core."""
    x = 0.0
    try:
        while True:
            x = (x * 1.0000001 + 3.14159) ** 0.5   # just enough to avoid optimisation
    except KeyboardInterrupt:
        pass


def monitor(processes):
    """Print live CPU % every second so you can see it climbing."""
    print("\n🔥  CPU STRESS ACTIVE — press Ctrl+C to stop\n")
    print(f"{'Time':>6}  {'CPU %':>6}  {'RAM %':>6}")
    print("-" * 26)
    elapsed = 0
    try:
        while True:
            cpu = psutil.cpu_percent(interval=1)
            ram = psutil.virtual_memory().percent
            bar = "█" * int(cpu / 5)
            print(f"{elapsed:>5}s  {cpu:>5.1f}%  {ram:>5.1f}%  {bar}")
            elapsed += 1
            if DURATION > 0 and elapsed >= DURATION:
                print(f"\n⏱  Auto-stopping after {DURATION}s.")
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nStopping stress processes...")
        for p in processes:
            p.terminate()
            p.join()


# ── Launch ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Ensure macOS multiprocessing support
    multiprocessing.set_start_method("spawn", force=True)
    
    processes = [multiprocessing.Process(target=burn_cpu, daemon=True) for _ in range(THREADS)]
    
    try:
        for p in processes:
            p.start()
        monitor(processes)
        print("\n✅  Stress stopped.  CPU will drop back to normal.")
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
        print("\n✅  Stress stopped.  CPU will drop back to normal.")