
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from queue import PriorityQueue
from threading import Lock
import json

from .base_agent import (
    BaseAgent, AgentStatus, AgentMessage, 
    MessageType, AgentLog
)


class MasterAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="MASTER_001",
            agent_name="Master Orchestrator",
            agent_type="master"
        )
        
        self.worker_agents: Dict[str, BaseAgent] = {}
        self.message_buffer: PriorityQueue = PriorityQueue()
        self.ueba_logs: List[AgentLog] = []
        self.workflow_state: Dict[str, Any] = {}
        self.lock = Lock()
        
        self.current_workflow = None
        self.workflow_history: List[Dict] = []
        
        self.anomaly_thresholds = {
            "rapid_actions": 10,  
            "unauthorized_receivers": ["EXTERNAL", "UNKNOWN"],
            "suspicious_patterns": []
        }
        
    def register_worker(self, worker: BaseAgent):
        self.worker_agents[worker.agent_id] = worker
        worker.set_master(self)
        self.log_action(
            action="register_worker",
            details={"worker_id": worker.agent_id, "worker_type": worker.agent_type}
        )
        
    def route_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        with self.lock:
            self.log_action(
                action="route_message",
                details={
                    "from": message.sender,
                    "to": message.receiver,
                    "type": message.message_type.value,
                    "priority": message.priority
                }
            )
            
            if message.receiver in self.anomaly_thresholds["unauthorized_receivers"]:
                self.trigger_security_alert(message)
                return None
                
            if message.receiver in self.worker_agents:
                target = self.worker_agents[message.receiver]
                target.receive_message(message)
                
                if message.requires_response:
                    response = target.process_message(message)
                    return response
                    
            return message
            
    def receive_log(self, log_entry: AgentLog):
        with self.lock:
            self.ueba_logs.append(log_entry)
            
            self.check_for_anomalies(log_entry)
            
    def check_for_anomalies(self, log_entry: AgentLog):
        recent_actions = [
            l for l in self.ueba_logs[-100:]
            if l.agent_id == log_entry.agent_id
            and (datetime.now() - l.timestamp).seconds < 1
        ]
        
        if len(recent_actions) > self.anomaly_thresholds["rapid_actions"]:
            self.trigger_security_alert(log_entry, "rapid_actions")
            
    def trigger_security_alert(self, source: Any, alert_type: str = "unauthorized"):
        alert = {
            "timestamp": datetime.now().isoformat(),
            "alert_type": alert_type,
            "source": str(source),
            "action": "blocked"
        }
        self.log_action(
            action="security_alert",
            details=alert,
            status="alert"
        )
        
        self.trigger_callback("security_alert", alert)
        
    def start_workflow(self, workflow_type: str, initial_data: Dict[str, Any]) -> str:
        workflow_id = f"WF_{int(time.time())}"
        
        self.current_workflow = {
            "workflow_id": workflow_id,
            "workflow_type": workflow_type,
            "started_at": datetime.now().isoformat(),
            "status": "active",
            "current_stage": "init",
            "data": initial_data,
            "stages_completed": []
        }
        
        self.log_action(
            action="start_workflow",
            details={"workflow_id": workflow_id, "type": workflow_type}
        )
        
        return workflow_id
        
    def update_workflow_stage(self, stage: str, stage_data: Dict[str, Any]):
        if self.current_workflow:
            self.current_workflow["stages_completed"].append({
                "stage": self.current_workflow["current_stage"],
                "completed_at": datetime.now().isoformat(),
                "data": stage_data
            })
            self.current_workflow["current_stage"] = stage
            self.current_workflow["data"].update(stage_data)
            
    def complete_workflow(self, final_data: Dict[str, Any] = None):
        if self.current_workflow:
            self.current_workflow["status"] = "completed"
            self.current_workflow["completed_at"] = datetime.now().isoformat()
            if final_data:
                self.current_workflow["data"].update(final_data)
            
            self.workflow_history.append(self.current_workflow)
            
            self.log_action(
                action="complete_workflow",
                details={"workflow_id": self.current_workflow["workflow_id"]}
            )
            
            completed = self.current_workflow
            self.current_workflow = None
            return completed
            
    def coordinate_fault_response(self, fault_data: Dict[str, Any]) -> Dict[str, Any]:
        workflow_id = self.start_workflow("fault_response", fault_data)
        
        result = {
            "workflow_id": workflow_id,
            "stages": {}
        }
        
        if "DATA_ANALYSIS_001" in self.worker_agents:
            analysis = self.worker_agents["DATA_ANALYSIS_001"].execute_task({
                "task": "analyze_telemetry",
                "data": fault_data
            })
            result["stages"]["data_analysis"] = analysis
            self.update_workflow_stage("diagnosis", analysis)
            
        if "DIAGNOSIS_001" in self.worker_agents:
            diagnosis = self.worker_agents["DIAGNOSIS_001"].execute_task({
                "task": "diagnose_fault",
                "data": result["stages"].get("data_analysis", fault_data)
            })
            result["stages"]["diagnosis"] = diagnosis
            self.update_workflow_stage("customer_engagement", diagnosis)
            
        if "CUSTOMER_ENGAGEMENT_001" in self.worker_agents:
            engagement = self.worker_agents["CUSTOMER_ENGAGEMENT_001"].execute_task({
                "task": "notify_customer",
                "data": result["stages"].get("diagnosis", {})
            })
            result["stages"]["customer_engagement"] = engagement
            self.update_workflow_stage("scheduling", engagement)
            
        if "SCHEDULING_001" in self.worker_agents:
            scheduling = self.worker_agents["SCHEDULING_001"].execute_task({
                "task": "book_tentative_slot",
                "data": result["stages"].get("diagnosis", {}),
                "customer_response": result["stages"].get("customer_engagement", {})
            })
            result["stages"]["scheduling"] = scheduling
            self.update_workflow_stage("manufacturing_insights", scheduling)
            
        if "MANUFACTURING_001" in self.worker_agents:
            insights = self.worker_agents["MANUFACTURING_001"].execute_task({
                "task": "log_for_capa",
                "data": result
            })
            result["stages"]["manufacturing_insights"] = insights
            
        completed = self.complete_workflow(result)
        result["workflow"] = completed
        
        return result
        
    def get_all_worker_status(self) -> Dict[str, Any]:
        return {
            agent_id: agent.get_status()
            for agent_id, agent in self.worker_agents.items()
        }
        
    def get_ueba_summary(self) -> Dict[str, Any]:
        now = datetime.now()
        recent_logs = [
            l for l in self.ueba_logs
            if (now - l.timestamp).seconds < 3600  
        ]
        
        return {
            "total_logs": len(self.ueba_logs),
            "recent_logs_1h": len(recent_logs),
            "agents_active": list(set(l.agent_id for l in recent_logs)),
            "alerts": [l.to_dict() for l in self.ueba_logs if l.status == "alert"]
        }
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.ALERT:
            self.log_action(
                action="handle_alert",
                details=message.payload
            )
            self.trigger_callback("alert_received", message.payload)
            
        elif message.message_type == MessageType.STATUS:
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=self.get_status()
            )
            
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        task_type = task.get("task")
        
        if task_type == "get_status":
            return self.get_all_worker_status()
        elif task_type == "get_ueba":
            return self.get_ueba_summary()
        elif task_type == "coordinate_fault":
            return self.coordinate_fault_response(task.get("data", {}))
            
        return {"error": "Unknown task type"}
