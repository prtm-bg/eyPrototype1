import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import sys
from datetime import datetime, timedelta
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from digital_twin import CausalDigitalTwin
from agents.master_agent import MasterAgent
from agents.worker_agents import (
    DataAnalysisAgent, DiagnosisAgent, CustomerEngagementAgent,
    SchedulingAgent, FeedbackAgent, ManufacturingInsightsAgent
)
from agents.voice_agent import VoiceAgent
from security.ueba import UEBASecurityLayer

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

st.set_page_config(
    page_title="Vehicle Digital Twin - Agentic AI",
    page_icon="car",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #667eea 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .agent-master { border-left-color: #e91e63; background: #fce4ec; }
    .agent-worker { border-left-color: #2196f3; background: #e3f2fd; }
    .agent-voice { border-left-color: #4caf50; background: #e8f5e9; }
    .metric-box {
        background: #f5f5f5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
    .fault-critical { background: #ffebee; border-left: 4px solid #f44336; }
    .fault-warning { background: #fff3e0; border-left: 4px solid #ff9800; }
    .fault-normal { background: #e8f5e9; border-left: 4px solid #4caf50; }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


def initialize_system():
    
    if 'initialized' not in st.session_state:
        st.session_state.digital_twin = CausalDigitalTwin()
        
        st.session_state.master_agent = MasterAgent()
        
        st.session_state.data_analysis_agent = DataAnalysisAgent()
        st.session_state.diagnosis_agent = DiagnosisAgent()
        st.session_state.customer_engagement_agent = CustomerEngagementAgent()
        st.session_state.scheduling_agent = SchedulingAgent()
        st.session_state.feedback_agent = FeedbackAgent()
        st.session_state.manufacturing_agent = ManufacturingInsightsAgent()
        
        st.session_state.voice_agent = VoiceAgent(output_dir="voice_output")
        
        st.session_state.ueba = UEBASecurityLayer()
        
        st.session_state.master_agent.register_worker(st.session_state.data_analysis_agent)
        st.session_state.master_agent.register_worker(st.session_state.diagnosis_agent)
        st.session_state.master_agent.register_worker(st.session_state.customer_engagement_agent)
        st.session_state.master_agent.register_worker(st.session_state.scheduling_agent)
        st.session_state.master_agent.register_worker(st.session_state.feedback_agent)
        st.session_state.master_agent.register_worker(st.session_state.manufacturing_agent)
        
        st.session_state.telemetry_data = None
        st.session_state.simulation_running = False
        st.session_state.current_frame = 0
        st.session_state.fault_detected = False
        st.session_state.current_analysis = None
        st.session_state.current_diagnosis = None
        st.session_state.current_scheduling = None
        st.session_state.appointment_confirmed = False
        st.session_state.voice_session = None
        st.session_state.agent_logs = []
        st.session_state.workflow_history = []
        st.session_state.loaded_file_key = None
        st.session_state.sim_step_size = 1
        
        st.session_state.initialized = True
        

def log_agent_action(agent_name: str, action: str, details: str):
    st.session_state.agent_logs.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "agent": agent_name,
        "action": action,
        "details": details
    })
    st.session_state.agent_logs = st.session_state.agent_logs[-50:]


def draw_causal_graph(graph_data: dict, highlight_component: str = None):
    
    G = nx.DiGraph()
    
    for node in graph_data["nodes"]:
        G.add_node(node["id"], **node)
        
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)
        
    fig, ax = plt.subplots(figsize=(12, 8))
    
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    category_colors = {
        "engine": "#ff6b6b",
        "electrical": "#4ecdc4",
        "braking": "#45b7d1",
        "cooling": "#96ceb4",
        "powertrain": "#ffeaa7",
        "interface": "#dfe6e9"
    }
    
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        node_data = G.nodes[node]
        category = node_data.get("category", "other")
        health = node_data.get("health_score", 100)
        
        if node == highlight_component:
            node_colors.append("#e74c3c")  
            node_sizes.append(3000)
        elif health < 50:
            node_colors.append("#e74c3c")  
            node_sizes.append(2000)
        elif health < 80:
            node_colors.append("#f39c12")  
            node_sizes.append(1500)
        else:
            node_colors.append(category_colors.get(category, "#bdc3c7"))
            node_sizes.append(1200)
            
    # Draw
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color="#95a5a6", arrows=True, 
                          arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)
    
    ax.set_title("Vehicle Component Causal Graph", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    return fig


def draw_simple_graph(highlight_node: str = None):
    
    G = nx.DiGraph()
    
    relationships = [
        ("Fuel System", "Engine", "feeds"),
        ("Ignition", "Engine", "ignites"),
        ("MAF Sensor", "ECU", "reports"),
        ("Crank Sensor", "ECU", "reports"),
        ("Wheel Speed FL", "ABS", "inputs"),
        ("Wheel Speed FR", "ABS", "inputs"),
        ("ECU", "Dashboard", "displays"),
        ("ABS", "Dashboard", "displays"),
        ("Cooling", "Engine", "cools"),
        ("HV Battery", "Motor", "powers")
    ]
    
    for u, v, w in relationships:
        G.add_edge(u, v, relationship=w)
        
    fig, ax = plt.subplots(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    for node in G.nodes():
        if node == highlight_node:
            node_colors.append("#e74c3c")
        elif "Sensor" in node:
            node_colors.append("#3498db")
        elif node in ["ECU", "ABS"]:
            node_colors.append("#9b59b6")
        elif node in ["Engine", "Motor"]:
            node_colors.append("#e67e22")
        else:
            node_colors.append("#2ecc71")
            
    nx.draw(G, pos, with_labels=True, node_color=node_colors,
            node_size=2000, font_size=8, font_weight="bold",
            arrows=True, edge_color="#7f8c8d", ax=ax)
    
    edge_labels = nx.get_edge_attributes(G, 'relationship')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
    
    ax.axis("off")
    return fig


def generate_voice_alert(text: str) -> str:
    
    if GTTS_AVAILABLE:
        try:
            os.makedirs("voice_output", exist_ok=True)
            filename = f"voice_output/alert_{int(time.time())}.mp3"
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            return filename
        except Exception as e:
            st.warning(f"Voice generation failed: {e}")
            return None
    return None


def run_fault_detection_workflow(telemetry: dict):
    
    master = st.session_state.master_agent
    
    log_agent_action("Digital Twin", "Processing", "Analyzing telemetry stream...")
    dt_result = st.session_state.digital_twin.process_telemetry(telemetry)
    st.session_state.current_analysis = dt_result
    
    if not dt_result["has_fault"]:
        return None
        
    st.session_state.ueba.log_action(
        "DIGITAL_TWIN",
        "fault_detected",
        {"dtc": dt_result["dtc_code"]}
    )
    
    log_agent_action("Data Analysis Agent", "Analyzing", "Detecting anomalies...")
    analysis_result = st.session_state.data_analysis_agent.execute_task({
        "task": "analyze_telemetry",
        "data": telemetry
    })
    st.session_state.ueba.log_action("DATA_ANALYSIS_001", "analyze_telemetry", {"anomalies": len(analysis_result.get("anomalies", []))})
    
    log_agent_action("Diagnosis Agent", "Diagnosing", "Performing root cause analysis...")
    diagnosis_result = st.session_state.diagnosis_agent.execute_task({
        "task": "diagnose_fault",
        "data": analysis_result
    })
    st.session_state.current_diagnosis = diagnosis_result
    st.session_state.ueba.log_action("DIAGNOSIS_001", "diagnose_fault", {"root_cause": diagnosis_result.get("root_cause")})
    
    log_agent_action("Scheduling Agent", "Booking", "Finding available service slots...")
    scheduling_result = st.session_state.scheduling_agent.execute_task({
        "task": "book_tentative_slot",
        "data": diagnosis_result
    })
    st.session_state.current_scheduling = scheduling_result
    st.session_state.ueba.log_action("SCHEDULING_001", "book_slot", {"slot": scheduling_result.get("date")})
    
    log_agent_action("Customer Engagement Agent", "Preparing", "Generating customer notification...")
    engagement_result = st.session_state.customer_engagement_agent.execute_task({
        "task": "notify_customer",
        "data": diagnosis_result
    })
    st.session_state.ueba.log_action("CUSTOMER_ENGAGEMENT_001", "notify_customer", {"severity": engagement_result.get("severity")})
    
    log_agent_action("Manufacturing Insights Agent", "Logging", "Recording for CAPA analysis...")
    st.session_state.manufacturing_agent.execute_task({
        "task": "log_for_capa",
        "data": {
            "workflow_id": f"WF_{int(time.time())}",
            "stages": {
                "diagnosis": diagnosis_result
            }
        }
    })
    
    workflow_result = {
        "timestamp": datetime.now().isoformat(),
        "analysis": analysis_result,
        "diagnosis": diagnosis_result,
        "scheduling": scheduling_result,
        "engagement": engagement_result,
        "dt_analysis": dt_result
    }
    
    st.session_state.workflow_history.append(workflow_result)
    
    return workflow_result


def main():
    
    initialize_system()
    
    st.markdown('<div class="main-header">Haloom Vehicle System :)</div>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("Upload Telemetry")
        
        uploaded_file = st.file_uploader(
            "Upload JSON or CSV telemetry file",
            type=['json', 'csv'],
            help="Upload vehicle telemetry data for simulation"
        )
        
        if uploaded_file is not None:
            file_key = f"{uploaded_file.name}_{uploaded_file.size}"
            
            if st.session_state.get('loaded_file_key') != file_key:
                try:
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        data = json.load(uploaded_file)
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                        else:
                            df = pd.DataFrame([data])
                            
                    # Preprocess
                    if 'dtc_code' in df.columns:
                        df['dtc_code'] = df['dtc_code'].fillna("None").astype(str)
                    else:
                        df['dtc_code'] = "None"
                        
                    st.session_state.telemetry_data = df
                    st.session_state.current_frame = 0
                    st.session_state.fault_detected = False
                    st.session_state.simulation_running = False
                    st.session_state.loaded_file_key = file_key
                    st.session_state.digital_twin.reset()
                    
                    st.success(f"Loaded {len(df)} telemetry frames")
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
            else:
                st.success(f"{len(st.session_state.telemetry_data)} frames loaded")
                
        st.divider()
        
        st.header("Simulation Controls")
        
        col1, col2 = st.columns(2)
        
        with col1:
            play_btn = st.button("Play", use_container_width=True, disabled=st.session_state.telemetry_data is None, key="play_btn")
        with col2:
            step_btn = st.button("Step", use_container_width=True, disabled=st.session_state.telemetry_data is None, key="step_btn")
            
        col3, col4 = st.columns(2)
        with col3:
            pause_btn = st.button("Pause", use_container_width=True, key="pause_btn")
        with col4:
            reset_btn = st.button("Reset", use_container_width=True, key="reset_btn")
        
        if play_btn:
            st.session_state.simulation_running = True
            st.session_state.fault_detected = False
            st.rerun()
            
        if pause_btn:
            st.session_state.simulation_running = False
            
        if step_btn and not st.session_state.fault_detected:
            if st.session_state.telemetry_data is not None and st.session_state.current_frame < len(st.session_state.telemetry_data):
                row = st.session_state.telemetry_data.iloc[st.session_state.current_frame].to_dict()
                dtc = str(row.get('dtc_code', 'None'))
                
                if dtc != 'None' and dtc != 'nan':
                    st.session_state.fault_detected = True
                    log_agent_action("Master Agent", "Orchestrating", "Fault detected - initiating response workflow")
                    run_fault_detection_workflow(row)
                else:
                    st.session_state.digital_twin.process_telemetry(row)
                    st.session_state.current_frame += 1
            
        if reset_btn:
            st.session_state.simulation_running = False
            st.session_state.current_frame = 0
            st.session_state.fault_detected = False
            st.session_state.current_analysis = None
            st.session_state.current_diagnosis = None
            st.session_state.current_scheduling = None
            st.session_state.appointment_confirmed = False
            st.session_state.digital_twin.reset()
            st.session_state.agent_logs = []
            st.rerun()
        
        st.divider()
        st.markdown("### Simulation Speed")
        if 'sim_step_size' not in st.session_state:
            st.session_state.sim_step_size = 1
        st.session_state.sim_step_size = st.slider("Frames per step", 1, 50, st.session_state.sim_step_size, help="Skip frames to speed up simulation")
        
        st.divider()
        st.markdown("### Simulation Status")
        if st.session_state.telemetry_data is not None:
            total_frames = len(st.session_state.telemetry_data)
            current = st.session_state.current_frame
            st.progress(current / total_frames if total_frames > 0 else 0)
            st.write(f"Frame: **{current}** / {total_frames}")
            
            if st.session_state.simulation_running:
                st.success("Simulation Running")
            elif st.session_state.fault_detected:
                st.error("Fault Detected!")
            else:
                st.info("Simulation Paused")
        else:
            st.write("Upload telemetry to start")
            
        st.divider()
        
        st.header("Security Status")
        security_summary = st.session_state.ueba.get_security_summary()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Alerts", security_summary["recent_alerts_1h"])
        with col2:
            st.metric("Blocked", len(security_summary["blocked_agents"]))
            
        if security_summary["alert_breakdown"]:
            for severity, count in security_summary["alert_breakdown"].items():
                if severity == "critical":
                    st.error(f"Critical: {count}")
                elif severity == "high":
                    st.warning(f"High: {count}")
                    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard", 
        "Causal System", 
        "Agents", 
        "Service", 
        "Analytics"
    ])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        if st.session_state.telemetry_data is not None and st.session_state.current_frame > 0:
            current_row = st.session_state.telemetry_data.iloc[min(st.session_state.current_frame-1, len(st.session_state.telemetry_data)-1)]
            
            with col1:
                speed = current_row.get('vehicle_speed_kmh', 0)
                st.metric("Speed", f"{speed:.1f} km/h")
                
            with col2:
                rpm = current_row.get('engine_rpm', 0)
                st.metric("RPM", f"{int(rpm)}")
                
            with col3:
                temp = current_row.get('engine_coolant_temp_c', 90)
                delta_color = "inverse" if temp > 100 else "off"
                st.metric("Engine Temp", f"{temp:.1f}°C", delta=f"{temp-90:.0f}°" if temp > 90 else None, delta_color=delta_color)
                
            with col4:
                dtc = current_row.get('dtc_code', 'None')
                mil = current_row.get('malfunction_indicator_lamp', 0)
                if str(dtc) not in ['None', 'nan', '']:
                    st.metric("DTC Code", str(dtc))
                elif mil == 1:
                    st.metric("MIL", "ON")
                else:
                    st.metric("Status", "Normal")
        
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                soc = current_row.get('hv_battery_soc_pct', 0)
                st.metric("Battery SOC", f"{soc:.1f}%")
            
            with col6:
                voltage = current_row.get('hv_battery_voltage_v', 0)
                st.metric("HV Voltage", f"{voltage:.0f}V")
            
            with col7:
                motor_temp = current_row.get('motor_temp_c', 0)
                st.metric("Motor Temp", f"{motor_temp:.1f}°C")
            
            with col8:
                inverter = current_row.get('inverter_temp_c', 0)
                st.metric("Inverter", f"{inverter:.1f}°C")
        
            col9, col10, col11, col12 = st.columns(4)
            
            with col9:
                throttle = current_row.get('throttle_position_pct', 0)
                st.metric("Throttle", f"{throttle:.0f}%")
            
            with col10:
                radar = current_row.get('radar_obj_count', 0)
                st.metric("Radar Objects", f"{int(radar)}")
            
            with col11:
                lane_conf = current_row.get('camera_lane_confidence', 0)
                st.metric("Lane Conf", f"{lane_conf:.0%}")
            
            with col12:
                autopilot = current_row.get('autopilot_engaged', 0)
                st.metric("Autopilot", "ON" if autopilot else "OFF")
                
        else:
            with col1:
                st.metric("Speed", "-- km/h")
            with col2:
                st.metric("RPM", "--")
            with col3:
                st.metric("Engine Temp", "--°C")
            with col4:
                st.metric("Status", "Waiting...")
                
        st.divider()
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            st.subheader("Vehicle Causal Graph")
            graph_placeholder = st.empty()
            
            highlight = None
            if st.session_state.current_diagnosis:
                highlight = st.session_state.current_diagnosis.get("root_cause")
                
            fig = draw_simple_graph(highlight_node=highlight)
            graph_placeholder.pyplot(fig)
            plt.close(fig)
            
        with col_right:
            st.subheader("Voice Agent")
            
            voice_container = st.container()
            
            with voice_container:
                if st.session_state.fault_detected and st.session_state.current_diagnosis:
                    diagnosis = st.session_state.current_diagnosis
                    scheduling = st.session_state.current_scheduling or {}
                    
                    severity = diagnosis.get("severity", "medium")
                    dtc = diagnosis.get("dtc_code", "UNKNOWN")
                    root_cause = diagnosis.get("root_cause", "potential issue")
                    slot_date = scheduling.get("date", "tomorrow")
                    slot_time = scheduling.get("time", "10:00 AM")
                    
                    voice_text = f"Attention. Fault code {dtc} detected. Analysis indicates {root_cause}. A service appointment has been tentatively booked for {slot_date} at {slot_time}. Please confirm or reschedule."
                    
                    st.error(f"**FAULT DETECTED: {dtc}**")
                    st.write(f"**Root Cause:** {root_cause}")
                    st.write(f"**Severity:** {severity.upper()}")
                    
                    audio_file = generate_voice_alert(voice_text)
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format='audio/mp3', autoplay=True)
                    
                    st.info(f"*\"{voice_text}\"*")
                    
                else:
                    st.success("System monitoring active. No faults detected.")
                    st.info("Voice agent on standby...")
                    
            if st.session_state.fault_detected and st.session_state.current_scheduling:
                st.divider()
                st.subheader("Service Appointment")
                
                scheduling = st.session_state.current_scheduling
                
                if not st.session_state.appointment_confirmed:
                    st.write(f"**Date:** {scheduling.get('date', 'Tomorrow')}")
                    st.write(f"**Time:** {scheduling.get('time', '10:00 AM')}")
                    st.write(f"**Location:** {scheduling.get('service_center_name', 'Service Center')}")
                    st.write(f"**Est. Cost:** ${scheduling.get('estimated_cost', 200)}")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("Confirm", use_container_width=True, type="primary"):
                            st.session_state.appointment_confirmed = True
                            log_agent_action("Scheduling Agent", "Confirmed", "Appointment confirmed by customer")
                            
                            response = st.session_state.customer_engagement_agent.process_customer_response({"action": "confirm"})
                            
                            confirm_text = "Excellent! Your appointment has been confirmed. You will receive a confirmation message with the service center details."
                            audio_file = generate_voice_alert(confirm_text)
                            if audio_file:
                                st.audio(audio_file, format='audio/mp3', autoplay=True)
                                
                            st.rerun()
                            
                    with col2:
                        if st.button("Reschedule", use_container_width=True):
                            st.session_state.show_reschedule = True
                            
                    with col3:
                        if st.button("Decline", use_container_width=True):
                            st.session_state.fault_detected = False
                            log_agent_action("Scheduling Agent", "Cancelled", "Appointment declined by customer")
                            st.rerun()
                            
                    # Show alternative slots
                    if getattr(st.session_state, 'show_reschedule', False):
                        st.write("**Alternative Slots:**")
                        alt_slots = scheduling.get('alternative_slots', [])
                        for i, slot in enumerate(alt_slots[:3]):
                            if st.button(f"{slot.get('date_display', slot.get('date', 'TBD'))} at {slot.get('time', 'TBD')}", key=f"slot_{i}"):
                                st.session_state.appointment_confirmed = True
                                log_agent_action("Scheduling Agent", "Rescheduled", f"Rescheduled to {slot.get('date')}")
                                st.rerun()
                else:
                    st.success("**Appointment Confirmed!**")
                    st.write(f"**Date:** {scheduling.get('date', 'Tomorrow')}")
                    st.write(f"**Time:** {scheduling.get('time', '10:00 AM')}")
                    st.write(f"**Location:** {scheduling.get('service_center_name', 'Service Center')}")
                    st.balloons()
                    
    with tab2:
        st.subheader("Causal Engine")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Temporal Transformer")
            st.write("Processes real-time sensor streams using attention mechanisms")
            
            if st.session_state.current_analysis:
                temporal = st.session_state.current_analysis.get("temporal_analysis", {})
                
                st.write("**Attention Scores:**")
                attention_scores = temporal.get("attention_scores", {})
                for sensor, score in list(attention_scores.items())[:5]:
                    st.progress(score, text=f"{sensor}: {score:.2f}")
                    
                if temporal.get("anomalies"):
                    st.error("**Detected Anomalies:**")
                    for anomaly in temporal["anomalies"]:
                        st.write(f"- {anomaly.get('sensor')}: {anomaly.get('type')} ({anomaly.get('severity')})")
                        
        with col2:
            st.markdown("#### Root Cause Analysis")
            
            if st.session_state.current_analysis:
                rca = st.session_state.current_analysis.get("root_cause_analysis", {})
                
                if rca.get("root_cause"):
                    st.error(f"**Root Cause:** {rca['root_cause']}")
                    st.write(f"**Confidence:** {rca.get('confidence', 0)*100:.0f}%")
                    
                    st.write("**Propagation Path:**")
                    path = rca.get("propagation_path", [])
                    if path:
                        st.write(" → ".join(path))
                        
                    st.write("**Affected Components:**")
                    for comp in rca.get("affected_components", [])[:5]:
                        st.write(f"- {comp}")
                else:
                    st.success("No faults detected in current analysis")
                    
        st.divider()
        
        st.markdown("#### Dynamic Causal Graph")
        
        if st.session_state.current_analysis:
            graph_data = st.session_state.current_analysis.get("graph_data", {})
            highlight = None
            if st.session_state.current_analysis.get("root_cause_analysis", {}).get("root_cause"):
                root_cause = st.session_state.current_analysis["root_cause_analysis"]["root_cause"]
                highlight = root_cause.lower().replace(" ", "_")
                
            fig = draw_causal_graph(graph_data, highlight_component=highlight)
            st.pyplot(fig)
            plt.close(fig)
        else:
            graph_data = st.session_state.digital_twin.causal_graph.to_visualization_data()
            fig = draw_causal_graph(graph_data)
            st.pyplot(fig)
            plt.close(fig)
            
    with tab3:
        st.subheader("Agentic AI Orchestration Layer")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### Agent Status")
            
            # Master Agent
            st.markdown("""
            <div class="agent-card agent-master">
                <strong>Master Agent</strong><br/>
                Main Orchestrator<br/>
                Status: <span style="color: green;">Active</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Worker Agents
            workers = [
                ("Data Analysis", st.session_state.data_analysis_agent),
                ("Diagnosis", st.session_state.diagnosis_agent),
                ("Customer Engagement", st.session_state.customer_engagement_agent),
                ("Scheduling", st.session_state.scheduling_agent),
                ("Feedback", st.session_state.feedback_agent),
                ("Manufacturing", st.session_state.manufacturing_agent),
            ]
            
            for name, agent in workers:
                status = agent.status.value
                st.markdown(f"""
                <div class="agent-card agent-worker">
                    <strong>{name} Agent</strong><br/>
                    Status: {status}
                </div>
                """, unsafe_allow_html=True)
                
            # Voice Agent
            st.markdown("""
            <div class="agent-card agent-voice">
                <strong>Voice Agent</strong><br/>
                Voice Communication<br/>
                Status: <span style="color: green;">Ready</span>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown("#### Agent Activity Log")
            
            if st.session_state.agent_logs:
                log_df = pd.DataFrame(st.session_state.agent_logs)
                st.dataframe(
                    log_df[["timestamp", "agent", "action", "details"]],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No agent activity yet. Start simulation to see agent interactions.")
                
            st.divider()
            

    with tab4:
        st.subheader("Service Operations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Available Service Slots")
            
            slots = st.session_state.scheduling_agent.get_available_slots()
            
            if slots:
                slot_df = pd.DataFrame(slots)
                slot_df = slot_df[slot_df['available'] == True][['date_display', 'time', 'service_center']].head(10)
                slot_df.columns = ['Date', 'Time', 'Center']
                st.dataframe(slot_df, use_container_width=True, hide_index=True)
            else:
                st.warning("No available slots")
                
        with col2:
            st.markdown("#### Booked Appointments")
            
            appointments = st.session_state.scheduling_agent.appointments
            
            if appointments:
                for apt_id, apt in appointments.items():
                    status_color = {
                        "tentative": "[TENT]",
                        "confirmed": "[CONF]",
                        "cancelled": "[CANC]",
                        "rescheduled": "[RSCH]"
                    }.get(apt.get("status", ""), "[----]")
                    
                    st.markdown(f"""
                    **{status_color} {apt_id}**
                    - Status: {apt.get('status', 'unknown').upper()}
                    - Date: {apt.get('slot', {}).get('date_display', 'TBD')}
                    - Time: {apt.get('slot', {}).get('time', 'TBD')}
                    - Severity: {apt.get('severity', 'N/A')}
                    """)
            else:
                st.info("No appointments booked yet")
                
        st.divider()
        
        st.markdown("#### Service Centers")
        
        centers = st.session_state.scheduling_agent.service_centers
        
        cols = st.columns(len(centers))
        for i, (center_id, center) in enumerate(centers.items()):
            with cols[i]:
                st.markdown(f"""
                **{center['name']}**
                - Location: {center['address']}
                - Capacity: {center['capacity']} bays
                - Specialties: {', '.join(center['specialties'])}
                """)
                
    with tab5:
        st.subheader("Manufacturing Feedback & Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Quality Insights Dashboard")
            
            insights = st.session_state.manufacturing_agent.get_quality_insights()
            
            st.metric("Total Faults Logged", insights.get("total_faults", 0))
            
            if insights.get("top_issues"):
                st.markdown("**Top Issues:**")
                for component, count in insights["top_issues"]:
                    st.progress(count / max(1, insights["total_faults"]), text=f"{component}: {count}")
                    
        with col2:
            st.markdown("#### UEBA Security Analytics")
            
            security = st.session_state.ueba.get_security_summary()
            
            st.metric("Actions Monitored", security.get("total_actions_logged", 0))
            st.metric("Agents Tracked", security.get("agents_monitored", 0))
            
            recent_alerts = st.session_state.ueba.get_recent_alerts(5)
            if recent_alerts:
                st.markdown("**Recent Security Alerts:**")
                for alert in recent_alerts:
                    severity_icon = {"critical": "[CRIT]", "high": "[HIGH]", "medium": "[MED]", "low": "[LOW]"}.get(alert["severity"], "[----]")
                    st.write(f"{severity_icon} {alert['description']}")
                    
        st.divider()
        
        st.markdown("#### Workflow History")
        
        if st.session_state.workflow_history:
            for i, workflow in enumerate(reversed(st.session_state.workflow_history[-5:])):
                with st.expander(f"Workflow {len(st.session_state.workflow_history) - i} - {workflow.get('timestamp', 'N/A')}"):
                    diagnosis = workflow.get("diagnosis", {})
                    st.write(f"**DTC:** {diagnosis.get('dtc_code', 'N/A')}")
                    st.write(f"**Root Cause:** {diagnosis.get('root_cause', 'N/A')}")
                    st.write(f"**Severity:** {diagnosis.get('severity', 'N/A')}")
                    st.write(f"**Reasoning:** {diagnosis.get('reasoning', 'N/A')}")
        else:
            st.info("No workflows completed yet. Run simulation to generate data.")
            
    if st.session_state.get('simulation_running', False) and st.session_state.get('telemetry_data') is not None:
        df = st.session_state.telemetry_data
        current = st.session_state.current_frame
        step_size = st.session_state.get('sim_step_size', 1)
        
        if current < len(df) and not st.session_state.get('fault_detected', False):
            fault_found = False
            fault_row = None
            fault_frame = current
            
            end_frame = min(current + step_size, len(df))
            for i in range(current, end_frame):
                row = df.iloc[i].to_dict()
                dtc = str(row.get('dtc_code', 'None'))
                
                if dtc not in ['None', 'nan', '', 'NaN', 'null', 'NaT']:
                    fault_found = True
                    fault_row = row
                    fault_frame = i
                    break
                    
            if fault_found:
                st.session_state.current_frame = fault_frame
                st.session_state.simulation_running = False
                st.session_state.fault_detected = True
                
                log_agent_action("Master Agent", "Orchestrating", f"Fault {fault_row.get('dtc_code')} detected at frame {fault_frame}")
                run_fault_detection_workflow(fault_row)
                st.rerun()
            else:
                last_row = df.iloc[end_frame - 1].to_dict()
                st.session_state.digital_twin.process_telemetry(last_row)
                st.session_state.current_frame = end_frame
                
                time.sleep(0.1)
                st.rerun()
        elif current >= len(df):
            st.session_state.simulation_running = False
            st.toast("Simulation complete - all frames processed!")


if __name__ == "__main__":
    main()
