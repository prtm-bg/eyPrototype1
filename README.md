# Vehicle Causal Engine - Agentic AI Prototype

A Streamlit-based prototype demonstrating predictive maintenance for vehicles using agentic AI, Graph based Causal Analysis Engine, and voice interaction.

## Features

- **Telemetry Upload**: Upload JSON or CSV vehicle telemetry data
- **Real-time Simulation**: Simulate vehicle operation with live metrics
- **Fault Detection**: Automatic detection of DTC (Diagnostic Trouble Codes)
- **Agentic Workflow**: Master-Worker agent orchestration for fault response
- **Causal Engine**: Causal graph visualization and root cause analysis
- **Voice Agent**: Text-to-speech notifications for detected faults
- **Service Booking**: Automatic appointment scheduling with confirmation

## Project Structure

```
prototype1/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── agents/                # Agent framework
│   ├── __init__.py
│   ├── base_agent.py      # Base agent class
│   ├── master_agent.py    # Master orchestrator agent
│   ├── worker_agents.py   # 6 specialized worker agents
│   └── voice_agent.py     # TTS voice agent
├── digital_twin/          # Digital twin engine
│   └── __init__.py        # Causal graph, transformers, GNN
├── security/              # Security layer
│   ├── __init__.py
│   └── ueba.py            # User/Entity Behavior Analytics
└── sample_data/           # Test telemetry files
    ├── sample_telemetry.json
    ├── sample_telemetry_misfire.json
    └── synthetic_vehicle_data_1765269906.csv
```

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Prototype

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage

1. Upload a telemetry file from `sample_data/` using the sidebar
2. Adjust simulation speed (frames per step) if using large CSV files
3. Click **Play** to start the simulation
4. Watch real-time metrics update on the Dashboard
5. When a fault is detected, the agent workflow triggers automatically
6. Confirm or reschedule the proposed service appointment

## Architecture

- **Master Agent**: Orchestrates the fault response workflow
- **Data Analysis Agent**: Analyzes telemetry for anomalies
- **Diagnosis Agent**: Determines root cause and severity
- **Customer Engagement Agent**: Manages customer communication
- **Scheduling Agent**: Books service appointments
- **Feedback Agent**: Processes customer feedback
- **Manufacturing Insights Agent**: Aggregates quality data
- **Voice Agent**: Provides audio notifications
- **UEBA Security Layer**: Monitors agent behavior for anomalies
