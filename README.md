# Intelligent AI Edge Analytics

This project simulates an Intelligent Edge Computing system. It utilizes a Reinforcement Learning agent (Proximal Policy Optimization - PPO) to make real-time decisions on whether to process computationally intensive tasks (like image processing) locally or offload them to an Edge Server. The AI optimizes for minimal execution time and energy consumption based on current CPU load, task complexity, and network latency.

## Project Structure

- `edge_server.py`: A FastAPI backend that acts as the remote "Edge Node" and performs heavy OpenCV image processing tasks.
- `train_agent.py`: A script to train the AI model using `stable-baselines3` in a simulated custom environment. It generates the `edge_ai_model.zip`.
- `edge_client.py`: A Streamlit web dashboard acting as the client device. It provides a UI to upload images, applies the AI model for offloading decisions, and visualizes real-time performance and sustainability metrics.
- `requirements.txt`: Core Python package dependencies.

## Setup Instructions

### 1. Install Dependencies
Ensure you have Python installed, then install the required packages. Open your terminal and run:

```bash
pip install -r requirements.txt
```

**Note:** The reinforcement learning components require `stable-baselines3`. If it was not installed, please install it manually:
```bash
pip install stable-baselines3[extra] gymnasium
```

### 2. Train the AI Model (Optional)
The project relies on a pre-trained `edge_ai_model.zip`. If it's missing or you wish to retrain the reinforcement learning model, run:
```bash
python train_agent.py
```
This will train the agent and save/overwrite the `edge_ai_model.zip` file in the directory.

### 3. Run the Edge Server
Start the backend FastAPI server that processes the offloaded image tasks. Run the following command:
```bash
uvicorn edge_server:app --reload --port 8000
```
The server will now be running at `http://localhost:8000`.

### 4. Run the Client Dashboard
Open a **new terminal window** (keep the edge server running in the first one) and launch the Streamlit application:
```bash
streamlit run edge_client.py
```
This will automatically open the web dashboard in your default browser (usually at `http://localhost:8501`).

## Using the Application

1. **Real-Time Processor Tab:**
   - **Upload an Image:** Choose a `.jpg` or `.png` file.
   - **Select a Filter:** Pick from Blur, Edge Detection, or Grayscale.
   - **Execute Task:** Click the button to process. The AI will analyze the live CPU load, task complexity, and simulated network latency to decide whether to process it on your machine ("Local") or send it to the server ("Remote").
   - **Override Controls:** You can use the sidebar to toggle "Override AI" and force the app to process everything locally (simulating a basic, non-intelligent app).

2. **Research Evaluation Tab:**
   - Switch to this tab after processing a few images to view the telemetry.
   - **Metrics & Graphs:** View the distribution of Local vs. Remote decisions and monitor execution speeds.
   - **Sustainability Model:** Compare the cumulative energy savings of the AI-Optimized Engine against a traditional "Always Local" baseline approach.
   - **Experiment Data Logs:** Access the raw dataset of your session's history for further analysis.