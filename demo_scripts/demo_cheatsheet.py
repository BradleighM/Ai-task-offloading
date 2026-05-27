"""
demo_cheatsheet.py  —  Run this to print your demo day command reference
"""

CHEATSHEET = """
╔══════════════════════════════════════════════════════════════════╗
║              DEMO DAY CHEAT SHEET                               ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  TERMINAL 1 — Start the edge server                             ║
║  ─────────────────────────────────────────────────────────────  ║
║  python latency_server.py                                        ║
║                                                                  ║
║  TERMINAL 2 — Start the Streamlit client                        ║
║  ─────────────────────────────────────────────────────────────  ║
║  streamlit run your_client_app.py                                ║
║                                                                  ║
║  TERMINAL 3 — Use this terminal for live switching              ║
║  ─────────────────────────────────────────────────────────────  ║
║                                                                  ║
║  SCENARIO A — Normal (agent stays LOCAL)                        ║
║    No stress, no delay. Just upload an image.                    ║
║    Expected: decision = LOCAL                                    ║
║                                                                  ║
║  SCENARIO B — High CPU, good network (agent OFFLOADS)          ║
║    Step 1: python cpu_stress.py          ← Terminal 4            ║
║    Step 2: upload image in Streamlit                             ║
║    Expected: decision = EDGE SERVER                              ║
║                                                                  ║
║  SCENARIO C — High CPU, slow network (agent stays LOCAL)       ║
║    Step 1: Keep cpu_stress.py running                            ║
║    Step 2 (run in Terminal 3):                                   ║
║      curl -X POST "http://localhost:8000/set-latency?mode=slow" ║
║    Step 3: upload image in Streamlit                             ║
║    Expected: decision = LOCAL  (smart! avoids slow network)     ║
║                                                                  ║
║  RESET between scenarios                                         ║
║    Stop stress:   Ctrl+C in Terminal 4                          ║
║    Reset network: curl -X POST "http://localhost:8000/           ║
║                   set-latency?mode=normal"                       ║
║                                                                  ║
║  JITTER demo (Recurrent PPO bonus)                              ║
║    curl -X POST "http://localhost:8000/set-latency?mode=jitter" ║
║                                                                  ║
║  CHECK what mode the server is in anytime                       ║
║    curl http://localhost:8000/latency-status                     ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║  KEY NUMBERS TO QUOTE TO LECTURERS                              ║
║  ─────────────────────────────────────────────────────────────  ║
║  Agent inference time:        4.2 ms   (target was < 10 ms)    ║
║  Total routing overhead:      8.6 ms                            ║
║  Scenario C — PPO agent:    121 ms                              ║
║  Scenario C — Always-Offload: 389 ms  (221% worse!)            ║
║  LSTM error reduction:        65.4%  fewer wrong decisions      ║
╚══════════════════════════════════════════════════════════════════╝
"""

print(CHEATSHEET)