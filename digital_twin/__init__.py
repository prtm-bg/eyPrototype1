

import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import json


@dataclass
class ComponentNode:
    component_id: str
    name: str
    category: str  # engine, electrical, braking, suspension, etc.
    health_score: float = 100.0
    fault_probability: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    fault_history: List[Dict] = field(default_factory=list)
    sensors: List[str] = field(default_factory=list)


@dataclass
class CausalEdge:
    source: str
    target: str
    relationship: str  # feeds, controls, monitors, inputs_to, etc.
    weight: float = 1.0  # Strength of causal influence
    delay_ms: float = 0  # Propagation delay


class TemporalTransformer:
    
    def __init__(self, window_size: int = 50, attention_heads: int = 4):
        self.window_size = window_size
        self.attention_heads = attention_heads
        self.sensor_buffers: Dict[str, deque] = {}
        self.attention_weights: Dict[str, np.ndarray] = {}
        
        # Key sensors to monitor
        self.key_sensors = [
            "engine_rpm", "vehicle_speed_kmh", "engine_coolant_temp_c",
            "mass_air_flow_gs", "throttle_position_pct", "oil_pressure_psi",
            "hv_battery_soc_pct", "hv_battery_voltage_v"
        ]
        
        # Initialize buffers
        for sensor in self.key_sensors:
            self.sensor_buffers[sensor] = deque(maxlen=window_size)
            
    def process_stream(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        
        results = {
            "timestamp": str(telemetry.get("timestamp", datetime.now())),
            "anomalies": [],
            "patterns": [],
            "attention_scores": {}
        }
        
        # Update buffers
        for sensor in self.key_sensors:
            if sensor in telemetry:
                value = telemetry[sensor]
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    self.sensor_buffers[sensor].append(float(value))
                    
        # Analyze each sensor
        for sensor, buffer in self.sensor_buffers.items():
            if len(buffer) >= 10:  # Need minimum data
                analysis = self._analyze_sensor_temporal(sensor, list(buffer))
                
                if analysis["anomaly_detected"]:
                    results["anomalies"].append({
                        "sensor": sensor,
                        "type": analysis["anomaly_type"],
                        "severity": analysis["severity"],
                        "current_value": buffer[-1] if buffer else None,
                        "expected_range": analysis["expected_range"]
                    })
                    
                if analysis["pattern_detected"]:
                    results["patterns"].append({
                        "sensor": sensor,
                        "pattern": analysis["pattern_type"],
                        "confidence": analysis["pattern_confidence"]
                    })
                    
                results["attention_scores"][sensor] = analysis["attention_score"]
                
        return results
        
    def _analyze_sensor_temporal(self, sensor: str, values: List[float]) -> Dict[str, Any]:
        
        result = {
            "anomaly_detected": False,
            "anomaly_type": None,
            "severity": "low",
            "pattern_detected": False,
            "pattern_type": None,
            "pattern_confidence": 0.0,
            "attention_score": 0.0,
            "expected_range": (0, 0)
        }
        
        if len(values) < 5:
            return result
            
        mean = np.mean(values)
        std = np.std(values)
        current = values[-1]
        recent = values[-5:]
        
        if std > 0:
            z_score = abs(current - mean) / std
            result["attention_score"] = min(1.0, z_score / 3.0)
        
        expected_ranges = {
            "engine_rpm": (600, 7000),
            "vehicle_speed_kmh": (0, 200),
            "engine_coolant_temp_c": (60, 105),
            "mass_air_flow_gs": (0.1, 100),
            "throttle_position_pct": (0, 100),
            "oil_pressure_psi": (15, 80),
            "hv_battery_soc_pct": (10, 100),
            "hv_battery_voltage_v": (300, 450)
        }
        
        expected = expected_ranges.get(sensor, (0, 1000))
        result["expected_range"] = expected
        
        if current < expected[0] or current > expected[1]:
            result["anomaly_detected"] = True
            result["anomaly_type"] = "out_of_range"
            result["severity"] = "high" if abs(current - mean) > 3 * std else "medium"
            
        if len(recent) >= 2:
            change_rate = abs(recent[-1] - recent[0]) / len(recent)
            normal_change = std * 0.5
            
            if change_rate > normal_change * 3:
                result["anomaly_detected"] = True
                result["anomaly_type"] = "sudden_change"
                result["severity"] = "high"
                
        if len(values) >= 20:
            crossings = 0
            for i in range(1, len(values)):
                if (values[i] - mean) * (values[i-1] - mean) < 0:
                    crossings += 1
                    
            oscillation_rate = crossings / len(values)
            if oscillation_rate > 0.4:
                result["pattern_detected"] = True
                result["pattern_type"] = "oscillation"
                result["pattern_confidence"] = min(1.0, oscillation_rate * 2)
                
            if len(values) >= 10:
                recent_trend = np.polyfit(range(10), values[-10:], 1)[0]
                if abs(recent_trend) > std * 0.1:
                    result["pattern_detected"] = True
                    result["pattern_type"] = "increasing" if recent_trend > 0 else "decreasing"
                    result["pattern_confidence"] = min(1.0, abs(recent_trend) / (std + 0.001))
                    
        return result


class SemanticTransformer:
    
    def __init__(self):
        self.dtc_semantics = {
            "P0300": {
                "system": "engine",
                "subsystem": "ignition",
                "severity": "high",
                "keywords": ["misfire", "spark", "coil", "ignition"],
                "related_codes": ["P0301", "P0302", "P0303", "P0304"]
            },
            "P0171": {
                "system": "engine",
                "subsystem": "fuel",
                "severity": "medium",
                "keywords": ["lean", "air", "fuel", "maf", "vacuum"],
                "related_codes": ["P0172", "P0174", "P0175"]
            },
            "C0035": {
                "system": "braking",
                "subsystem": "abs",
                "severity": "high",
                "keywords": ["wheel", "speed", "sensor", "abs"],
                "related_codes": ["C0036", "C0037", "C0038"]
            },
            "U0100": {
                "system": "electrical",
                "subsystem": "communication",
                "severity": "critical",
                "keywords": ["communication", "ecu", "pcm", "can", "bus"],
                "related_codes": ["U0101", "U0102", "U0103"]
            }
        }
        
        self.action_semantics = {
            "replace": {"impact": 0.9, "cost_factor": 1.0},
            "repair": {"impact": 0.7, "cost_factor": 0.6},
            "clean": {"impact": 0.5, "cost_factor": 0.2},
            "adjust": {"impact": 0.4, "cost_factor": 0.1},
            "inspect": {"impact": 0.2, "cost_factor": 0.1}
        }
        
    def process_log(self, log_entry: Dict[str, Any]) -> Dict[str, Any]:
        
        result = {
            "log_id": log_entry.get("id", "unknown"),
            "systems_affected": [],
            "severity_score": 0.0,
            "semantic_features": {},
            "recommendations": []
        }
        
        dtc = log_entry.get("dtc_code", "")
        description = log_entry.get("description", "").lower()
        action_taken = log_entry.get("action", "").lower()
        
        if dtc in self.dtc_semantics:
            semantics = self.dtc_semantics[dtc]
            result["systems_affected"].append(semantics["system"])
            result["semantic_features"]["subsystem"] = semantics["subsystem"]
            result["semantic_features"]["keywords_matched"] = [
                kw for kw in semantics["keywords"] if kw in description
            ]
            result["severity_score"] = {
                "critical": 1.0, "high": 0.8, "medium": 0.5, "low": 0.2
            }.get(semantics["severity"], 0.3)
            
        for action, props in self.action_semantics.items():
            if action in action_taken:
                result["semantic_features"]["action_type"] = action
                result["semantic_features"]["action_impact"] = props["impact"]
                break
                
        return result
        
    def extract_patterns(self, logs: List[Dict]) -> Dict[str, Any]:
        
        patterns = {
            "recurring_issues": {},
            "system_health": {},
            "trending_components": []
        }
        
        for log in logs:
            processed = self.process_log(log)
            for system in processed["systems_affected"]:
                patterns["recurring_issues"][system] = patterns["recurring_issues"].get(system, 0) + 1
                
        return patterns


class DynamicCausalGraph:
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.components: Dict[str, ComponentNode] = {}
        self._build_initial_topology()
        
    def _build_initial_topology(self):
        
        components = [
            ComponentNode("fuel_pump", "Fuel Pump", "engine", sensors=["fuel_pressure"]),
            ComponentNode("fuel_injectors", "Fuel Injectors", "engine", sensors=["fuel_rate"]),
            ComponentNode("spark_plugs", "Spark Plugs", "engine", sensors=["misfire_count"]),
            ComponentNode("ignition_coils", "Ignition Coils", "engine", sensors=["coil_primary_v"]),
            ComponentNode("maf_sensor", "MAF Sensor", "engine", sensors=["mass_air_flow_gs"]),
            ComponentNode("throttle_body", "Throttle Body", "engine", sensors=["throttle_position_pct"]),
            ComponentNode("engine", "Engine", "engine", sensors=["engine_rpm", "engine_load_pct"]),
            ComponentNode("ecu", "ECU", "electrical", sensors=["dtc_code"]),
            
            ComponentNode("radiator", "Radiator", "cooling", sensors=["coolant_temp"]),
            ComponentNode("thermostat", "Thermostat", "cooling", sensors=["engine_coolant_temp_c"]),
            ComponentNode("water_pump", "Water Pump", "cooling", sensors=["coolant_flow"]),
            
            ComponentNode("wheel_speed_fl", "Wheel Speed FL", "braking", sensors=["wheel_speed_fl"]),
            ComponentNode("wheel_speed_fr", "Wheel Speed FR", "braking", sensors=["wheel_speed_fr"]),
            ComponentNode("wheel_speed_rl", "Wheel Speed RL", "braking", sensors=["wheel_speed_rl"]),
            ComponentNode("wheel_speed_rr", "Wheel Speed RR", "braking", sensors=["wheel_speed_rr"]),
            ComponentNode("abs_module", "ABS Module", "braking", sensors=["abs_active"]),
            ComponentNode("brake_master", "Brake Master Cylinder", "braking", sensors=["brake_pressure_bar"]),
            
            ComponentNode("battery_12v", "12V Battery", "electrical", sensors=["battery_voltage"]),
            ComponentNode("alternator", "Alternator", "electrical", sensors=["charging_rate"]),
            ComponentNode("can_bus", "CAN Bus", "electrical", sensors=["can_errors"]),
            
            ComponentNode("hv_battery", "HV Battery", "electrical", sensors=["hv_battery_soc_pct", "hv_battery_voltage_v"]),
            ComponentNode("inverter", "Inverter", "electrical", sensors=["inverter_temp_c"]),
            ComponentNode("electric_motor", "Electric Motor", "powertrain", sensors=["motor_temp_c"]),
            
            ComponentNode("transmission", "Transmission", "powertrain", sensors=["gear_position"]),
            ComponentNode("crankshaft_sensor", "Crankshaft Sensor", "engine", sensors=["crank_position"]),
            
            ComponentNode("dashboard", "Dashboard", "interface", sensors=["malfunction_indicator_lamp"]),
        ]
        
        for component in components:
            self.components[component.component_id] = component
            self.graph.add_node(component.component_id, **vars(component))
            
        edges = [
            CausalEdge("fuel_pump", "fuel_injectors", "feeds", 0.9),
            CausalEdge("fuel_injectors", "engine", "supplies", 0.95),
            
            CausalEdge("ignition_coils", "spark_plugs", "powers", 0.95),
            CausalEdge("spark_plugs", "engine", "ignites", 0.9),
            
            CausalEdge("maf_sensor", "ecu", "reports_to", 0.8),
            CausalEdge("throttle_body", "engine", "controls_air", 0.85),
            
            CausalEdge("ecu", "fuel_injectors", "controls", 0.95),
            CausalEdge("ecu", "ignition_coils", "controls", 0.95),
            CausalEdge("ecu", "throttle_body", "controls", 0.9),
            CausalEdge("crankshaft_sensor", "ecu", "reports_to", 0.9),
            
            CausalEdge("water_pump", "radiator", "circulates", 0.9),
            CausalEdge("thermostat", "radiator", "regulates", 0.8),
            CausalEdge("radiator", "engine", "cools", 0.85),
            
            CausalEdge("wheel_speed_fl", "abs_module", "inputs", 0.95),
            CausalEdge("wheel_speed_fr", "abs_module", "inputs", 0.95),
            CausalEdge("wheel_speed_rl", "abs_module", "inputs", 0.95),
            CausalEdge("wheel_speed_rr", "abs_module", "inputs", 0.95),
            CausalEdge("brake_master", "abs_module", "feeds", 0.9),
            
            CausalEdge("battery_12v", "ecu", "powers", 0.95),
            CausalEdge("alternator", "battery_12v", "charges", 0.9),
            CausalEdge("can_bus", "ecu", "connects", 0.95),
            
            CausalEdge("hv_battery", "inverter", "powers", 0.95),
            CausalEdge("inverter", "electric_motor", "drives", 0.95),
            CausalEdge("electric_motor", "transmission", "powers", 0.9),
            
            CausalEdge("ecu", "dashboard", "communicates", 0.95),
            CausalEdge("abs_module", "dashboard", "communicates", 0.9),
        ]
        
        for edge in edges:
            self.graph.add_edge(
                edge.source, edge.target,
                relationship=edge.relationship,
                weight=edge.weight,
                delay_ms=edge.delay_ms
            )
            
    def update_component_health(self, component_id: str, health_delta: float, fault_info: Dict = None):
        
        if component_id in self.components:
            component = self.components[component_id]
            component.health_score = max(0, min(100, component.health_score + health_delta))
            component.last_updated = datetime.now()
            
            if fault_info:
                component.fault_history.append({
                    "timestamp": datetime.now().isoformat(),
                    **fault_info
                })
                
            self.graph.nodes[component_id]["health_score"] = component.health_score
            
    def get_component_by_dtc(self, dtc: str) -> Optional[str]:
        
        dtc_component_map = {
            "P0300": "spark_plugs",
            "P0171": "maf_sensor",
            "P0172": "fuel_injectors",
            "C0035": "wheel_speed_fl",
            "C0036": "wheel_speed_fr",
            "U0100": "can_bus",
            "P0128": "thermostat",
            "P0420": "engine",
            "P0455": "fuel_pump"
        }
        
        return dtc_component_map.get(dtc)
        
    def get_affected_components(self, source_component: str, depth: int = 2) -> List[str]:
        
        affected = []
        
        try:
            descendants = nx.descendants_at_distance(self.graph, source_component, depth)
            affected.extend(descendants)
        except:
            pass
            
        try:
            ancestors = nx.ancestors(self.graph)
            for node in self.graph.predecessors(source_component):
                affected.append(node)
        except:
            pass
            
        return list(set(affected))
        
    def get_propagation_path(self, source: str, target: str) -> List[str]:
        
        try:
            path = nx.shortest_path(self.graph, source, target)
            return path
        except nx.NetworkXNoPath:
            return []
            
    def to_visualization_data(self) -> Dict[str, Any]:
        
        nodes = []
        edges = []
        
        for node_id, data in self.graph.nodes(data=True):
            component = self.components.get(node_id)
            nodes.append({
                "id": node_id,
                "name": component.name if component else node_id,
                "category": component.category if component else "unknown",
                "health_score": component.health_score if component else 100,
                "fault_probability": component.fault_probability if component else 0
            })
            
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "relationship": data.get("relationship", "connected"),
                "weight": data.get("weight", 1.0)
            })
            
        return {"nodes": nodes, "edges": edges}


class GNNPropagationEngine:
    
    def __init__(self, causal_graph: DynamicCausalGraph):
        self.causal_graph = causal_graph
        self.propagation_iterations = 3
        self.decay_factor = 0.7
        
    def propagate_fault(self, source_component: str, fault_severity: float) -> Dict[str, float]:
        
        graph = self.causal_graph.graph
        probabilities = {node: 0.0 for node in graph.nodes()}
        probabilities[source_component] = fault_severity
        
        for iteration in range(self.propagation_iterations):
            new_probabilities = probabilities.copy()
            
            for node in graph.nodes():
                if probabilities[node] > 0:
                    for neighbor in graph.successors(node):
                        edge_data = graph.get_edge_data(node, neighbor)
                        weight = edge_data.get("weight", 1.0)
                        
                        propagated = probabilities[node] * weight * self.decay_factor
                        new_probabilities[neighbor] = max(
                            new_probabilities[neighbor],
                            propagated
                        )
                        
            probabilities = new_probabilities
            
        affected = {k: round(v, 3) for k, v in probabilities.items() if v > 0.05}
        
        return affected
        
    def calculate_system_risk(self, component_faults: Dict[str, float]) -> Dict[str, Any]:
        
        system_risks = {}
        
        for comp_id, fault_prob in component_faults.items():
            if comp_id in self.causal_graph.components:
                category = self.causal_graph.components[comp_id].category
                if category not in system_risks:
                    system_risks[category] = []
                system_risks[category].append(fault_prob)
                
        system_scores = {}
        for system, risks in system_risks.items():
            system_scores[system] = max(risks) if risks else 0
            
        overall_risk = max(system_scores.values()) if system_scores else 0
        
        return {
            "overall_risk": round(overall_risk, 3),
            "system_risks": system_scores,
            "component_risks": component_faults
        }


class RootCauseAnalyzer:
    
    def __init__(self, causal_graph: DynamicCausalGraph, propagation_engine: GNNPropagationEngine):
        self.causal_graph = causal_graph
        self.propagation_engine = propagation_engine
        
    def analyze(self, telemetry: Dict[str, Any], temporal_analysis: Dict[str, Any]) -> Dict[str, Any]:
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "root_cause": None,
            "confidence": 0.0,
            "affected_components": [],
            "propagation_path": [],
            "health_status": {},
            "recommendations": []
        }
        
        dtc = str(telemetry.get("dtc_code", "None"))
        
        primary_component = self.causal_graph.get_component_by_dtc(dtc)
        
        if primary_component:
            result["root_cause"] = primary_component
            
            fault_severity = 0.9 if dtc != "None" else 0.5
            propagation = self.propagation_engine.propagate_fault(primary_component, fault_severity)
            
            result["affected_components"] = list(propagation.keys())
            
            if "dashboard" in propagation:
                result["propagation_path"] = self.causal_graph.get_propagation_path(
                    primary_component, "dashboard"
                )
                
            for comp_id, fault_prob in propagation.items():
                self.causal_graph.update_component_health(
                    comp_id, 
                    -fault_prob * 20,  # Health decrease proportional to fault probability
                    {"dtc": dtc, "probability": fault_prob}
                )
                
            temporal_anomalies = temporal_analysis.get("anomalies", [])
            matching_anomalies = [
                a for a in temporal_anomalies
                if primary_component in self.causal_graph.components and
                any(s in a.get("sensor", "") for s in self.causal_graph.components[primary_component].sensors)
            ]
            
            if matching_anomalies:
                result["confidence"] = min(0.95, 0.7 + 0.1 * len(matching_anomalies))
            else:
                result["confidence"] = 0.7
                
        else:
            anomalies = temporal_analysis.get("anomalies", [])
            if anomalies:
                severe = max(anomalies, key=lambda x: {"high": 3, "medium": 2, "low": 1}.get(x.get("severity", "low"), 0))
                sensor = severe.get("sensor", "")
                
                for comp_id, comp in self.causal_graph.components.items():
                    if sensor in comp.sensors:
                        result["root_cause"] = comp_id
                        result["confidence"] = 0.5
                        break
                        
        for comp_id, comp in self.causal_graph.components.items():
            result["health_status"][comp_id] = {
                "name": comp.name,
                "health_score": round(comp.health_score, 1),
                "category": comp.category
            }
            
        if result["root_cause"]:
            component = self.causal_graph.components.get(result["root_cause"])
            if component:
                result["recommendations"].append(f"Inspect {component.name}")
                result["recommendations"].append(f"Check related {component.category} system components")
                
        return result


class CausalDigitalTwin:
    
    def __init__(self):
        self.temporal_transformer = TemporalTransformer()
        self.semantic_transformer = SemanticTransformer()
        self.causal_graph = DynamicCausalGraph()
        self.propagation_engine = GNNPropagationEngine(self.causal_graph)
        self.root_cause_analyzer = RootCauseAnalyzer(self.causal_graph, self.propagation_engine)
        
        self.telemetry_history: List[Dict] = []
        self.analysis_history: List[Dict] = []
        
    def process_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        
        temporal_result = self.temporal_transformer.process_stream(telemetry)
        
        rca_result = self.root_cause_analyzer.analyze(telemetry, temporal_result)
        
        component_faults = {}
        for comp_id in rca_result["affected_components"]:
            comp = self.causal_graph.components.get(comp_id)
            if comp:
                component_faults[comp_id] = (100 - comp.health_score) / 100
                
        risk_analysis = self.propagation_engine.calculate_system_risk(component_faults)
        
        result = {
            "timestamp": telemetry.get("timestamp", datetime.now().isoformat()),
            "vin": telemetry.get("vin", "UNKNOWN"),
            "dtc_code": str(telemetry.get("dtc_code", "None")),
            "temporal_analysis": temporal_result,
            "root_cause_analysis": rca_result,
            "risk_analysis": risk_analysis,
            "has_fault": rca_result["root_cause"] is not None,
            "graph_data": self.causal_graph.to_visualization_data()
        }
        
        self.telemetry_history.append(telemetry)
        self.analysis_history.append(result)
        
        return result
        
    def get_component_health(self) -> Dict[str, Any]:
        
        return {
            comp_id: {
                "name": comp.name,
                "category": comp.category,
                "health_score": round(comp.health_score, 1),
                "fault_probability": round(comp.fault_probability, 3)
            }
            for comp_id, comp in self.causal_graph.components.items()
        }
        
    def get_graph_for_visualization(self, highlight_component: str = None) -> Dict[str, Any]:
        
        data = self.causal_graph.to_visualization_data()
        
        if highlight_component:
            for node in data["nodes"]:
                node["highlighted"] = node["id"] == highlight_component
                
        return data
        
    def reset(self):
        
        self.temporal_transformer = TemporalTransformer()
        self.causal_graph = DynamicCausalGraph()
        self.propagation_engine = GNNPropagationEngine(self.causal_graph)
        self.root_cause_analyzer = RootCauseAnalyzer(self.causal_graph, self.propagation_engine)
        self.telemetry_history.clear()
        self.analysis_history.clear()
