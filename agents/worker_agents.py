
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

from .base_agent import (
    BaseAgent, AgentStatus, AgentMessage, 
    MessageType, AgentLog
)


class DataAnalysisAgent(BaseAgent):

    def __init__(self):
        super().__init__(
            agent_id="DATA_ANALYSIS_001",
            agent_name="Data Analysis Agent",
            agent_type="worker"
        )
        
        self.thresholds = {
            "engine_temp_high": 105,
            "engine_temp_critical": 120,
            "rpm_max": 7000,
            "oil_pressure_min": 15,
            "battery_soc_low": 20,
            "maf_min": 0.2,
            "speed_rpm_ratio_max": 0.02  
        }
        
        self.history_buffer: List[Dict] = []
        self.buffer_size = 100
        
    def analyze_telemetry(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        anomalies = []
        warnings = []
        health_score = 100
        
        dtc = str(telemetry.get("dtc_code", "None"))
        if dtc != "None" and dtc != "nan":
            anomalies.append({
                "type": "dtc_code",
                "code": dtc,
                "severity": "critical",
                "description": self.get_dtc_description(dtc)
            })
            health_score -= 40
            
        engine_temp = telemetry.get("engine_coolant_temp_c", 90)
        if engine_temp > self.thresholds["engine_temp_critical"]:
            anomalies.append({
                "type": "engine_overheating",
                "value": engine_temp,
                "threshold": self.thresholds["engine_temp_critical"],
                "severity": "critical"
            })
            health_score -= 30
        elif engine_temp > self.thresholds["engine_temp_high"]:
            warnings.append({
                "type": "engine_temp_elevated",
                "value": engine_temp,
                "threshold": self.thresholds["engine_temp_high"],
                "severity": "warning"
            })
            health_score -= 10
            
        rpm = telemetry.get("engine_rpm", 0)
        if rpm > self.thresholds["rpm_max"]:
            warnings.append({
                "type": "rpm_high",
                "value": rpm,
                "threshold": self.thresholds["rpm_max"],
                "severity": "warning"
            })
            health_score -= 10
            
        maf = telemetry.get("mass_air_flow_gs", 1.0)
        speed = telemetry.get("vehicle_speed_kmh", 0)
        if speed > 10 and maf < self.thresholds["maf_min"]:
            anomalies.append({
                "type": "maf_sensor_fault",
                "value": maf,
                "threshold": self.thresholds["maf_min"],
                "severity": "critical"
            })
            health_score -= 25
            
        if speed > 10 and rpm == 0:
            anomalies.append({
                "type": "engine_stall",
                "description": "Vehicle moving but engine RPM is 0",
                "severity": "critical"
            })
            health_score -= 50
            
        if "hv_battery_soc_pct" in telemetry:
            soc = telemetry["hv_battery_soc_pct"]
            if soc < self.thresholds["battery_soc_low"]:
                warnings.append({
                    "type": "battery_low",
                    "value": soc,
                    "threshold": self.thresholds["battery_soc_low"],
                    "severity": "warning"
                })
                health_score -= 15
                
        health_score = max(0, health_score)
        
        result = {
            "timestamp": str(telemetry.get("timestamp", datetime.now())),
            "vin": telemetry.get("vin", "UNKNOWN"),
            "health_score": health_score,
            "anomalies": anomalies,
            "warnings": warnings,
            "has_fault": len(anomalies) > 0,
            "dtc_code": dtc if dtc != "None" else None,
            "raw_telemetry": telemetry
        }
        
        self.log_action(
            action="analyze_telemetry",
            details={
                "health_score": health_score,
                "anomaly_count": len(anomalies),
                "warning_count": len(warnings)
            }
        )
        
        return result
        
    def get_dtc_description(self, dtc: str) -> str:
        dtc_map = {
            "P0300": "Random/Multiple Cylinder Misfire Detected",
            "P0171": "System Too Lean (Bank 1)",
            "P0172": "System Too Rich (Bank 1)",
            "C0035": "Left Front Wheel Speed Sensor Circuit",
            "U0100": "Lost Communication With ECM/PCM",
            "P0420": "Catalyst System Efficiency Below Threshold",
            "P0455": "Evaporative Emission System Leak Detected (Large Leak)",
            "P0128": "Coolant Thermostat (Coolant Temperature Below Regulating)",
            "B1234": "Airbag Module Communication Error",
            "C1234": "ABS Hydraulic Pump Motor Circuit"
        }
        return dtc_map.get(dtc, f"Unknown fault code: {dtc}")
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            telemetry = message.payload.get("telemetry")
            if telemetry:
                result = self.analyze_telemetry(telemetry)
                return AgentMessage(
                    sender=self.agent_id,
                    receiver=message.sender,
                    message_type=MessageType.RESPONSE,
                    payload=result
                )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "analyze_telemetry":
            data = task.get("data", {})
            result = self.analyze_telemetry(data)
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}


class DiagnosisAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="DIAGNOSIS_001",
            agent_name="Diagnosis Agent",
            agent_type="worker"
        )
        
        self.causal_knowledge = {
            "P0300": {
                "root_causes": ["Spark Plugs", "Ignition Coils", "Fuel Injectors", "Crankshaft Sensor"],
                "primary_suspect": "Spark Plugs",
                "propagation_path": ["Spark Plugs", "Ignition System", "Engine", "Performance"],
                "severity": "high",
                "urgency": "schedule_soon"
            },
            "P0171": {
                "root_causes": ["MAF Sensor", "Vacuum Leak", "Fuel Pump", "Oxygen Sensor"],
                "primary_suspect": "MAF Sensor",
                "propagation_path": ["MAF Sensor", "Air Intake", "Fuel System", "Engine"],
                "severity": "medium",
                "urgency": "schedule_soon"
            },
            "C0035": {
                "root_causes": ["Wheel Speed Sensor FL", "Wiring Harness", "ABS Module"],
                "primary_suspect": "Wheel Speed Sensor FL",
                "propagation_path": ["Wheel Speed Sensor", "ABS Module", "Braking System"],
                "severity": "high",
                "urgency": "immediate"
            },
            "U0100": {
                "root_causes": ["CAN Bus", "ECU", "Wiring", "Ground Connection"],
                "primary_suspect": "CAN Bus Communication",
                "propagation_path": ["CAN Bus", "ECU", "All Systems"],
                "severity": "critical",
                "urgency": "immediate"
            }
        }
        
        self.failure_models = {
            "Spark Plugs": {"mtbf_days": 365, "replacement_cost": 150},
            "MAF Sensor": {"mtbf_days": 730, "replacement_cost": 250},
            "Wheel Speed Sensor FL": {"mtbf_days": 1095, "replacement_cost": 180},
            "CAN Bus Communication": {"mtbf_days": 1825, "replacement_cost": 500}
        }
        
    def diagnose_fault(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        
        dtc = analysis_data.get("dtc_code") or analysis_data.get("data", {}).get("dtc_code")
        anomalies = analysis_data.get("anomalies", [])
        raw_telemetry = analysis_data.get("raw_telemetry", {})
        
        diagnosis = {
            "timestamp": datetime.now().isoformat(),
            "dtc_code": dtc,
            "root_cause": None,
            "confidence": 0,
            "reasoning": "",
            "propagation_path": [],
            "severity": "unknown",
            "urgency": "unknown",
            "recommended_action": "",
            "estimated_repair_cost": 0,
            "failure_probability_30d": 0
        }
        
        if dtc and dtc in self.causal_knowledge:
            kb = self.causal_knowledge[dtc]
            
            root_cause = self.apply_causal_reasoning(dtc, raw_telemetry)
            
            diagnosis.update({
                "root_cause": root_cause,
                "confidence": 0.85,
                "reasoning": self.generate_reasoning(dtc, root_cause, raw_telemetry),
                "propagation_path": kb["propagation_path"],
                "severity": kb["severity"],
                "urgency": kb["urgency"],
                "recommended_action": f"Inspect and replace {root_cause}",
                "estimated_repair_cost": self.failure_models.get(root_cause, {}).get("replacement_cost", 200),
                "failure_probability_30d": self.calculate_failure_probability(root_cause)
            })
        else:
            diagnosis.update({
                "root_cause": "Unknown Component",
                "confidence": 0.5,
                "reasoning": "Unable to determine exact cause. Recommend professional diagnosis.",
                "severity": "medium",
                "urgency": "schedule_soon",
                "recommended_action": "Visit service center for diagnostic scan"
            })
            
        self.log_action(
            action="diagnose_fault",
            details={
                "dtc": dtc,
                "root_cause": diagnosis["root_cause"],
                "severity": diagnosis["severity"]
            }
        )
        
        return diagnosis
        
    def apply_causal_reasoning(self, dtc: str, telemetry: Dict) -> str:
        
        kb = self.causal_knowledge.get(dtc, {})
        candidates = kb.get("root_causes", [])
        
        if dtc == "P0300":
            rpm = telemetry.get("engine_rpm", 800)
            if rpm < 100:
                return "Crankshaft Sensor"
            return "Spark Plugs"
            
        elif dtc == "P0171":
            maf = telemetry.get("mass_air_flow_gs", 1.0)
            if maf < 0.2:
                return "MAF Sensor"
            return "Vacuum Leak"
            
        elif dtc == "C0035":
            return "Wheel Speed Sensor FL"
            
        elif dtc == "U0100":
            return "CAN Bus Communication"
            
        return kb.get("primary_suspect", "Unknown")
        
    def generate_reasoning(self, dtc: str, root_cause: str, telemetry: Dict) -> str:
        
        reasoning_templates = {
            "P0300": f"DTC {dtc} indicates random misfire. Analysis of engine RPM pattern ({telemetry.get('engine_rpm', 'N/A')} RPM) and throttle position suggests ignition system fault. Most likely cause: {root_cause}.",
            "P0171": f"DTC {dtc} indicates lean air-fuel mixture. MAF sensor reading ({telemetry.get('mass_air_flow_gs', 'N/A')} g/s) is below normal threshold while vehicle is in motion. Primary suspect: {root_cause}.",
            "C0035": f"DTC {dtc} indicates wheel speed sensor circuit fault. Vehicle speed ({telemetry.get('vehicle_speed_kmh', 'N/A')} km/h) detected but sensor reports zero. Likely cause: {root_cause}.",
            "U0100": f"DTC {dtc} indicates lost communication with ECM/PCM. This affects multiple systems. CAN bus or module failure suspected. Primary suspect: {root_cause}."
        }
        
        return reasoning_templates.get(dtc, f"Fault code {dtc} detected. Root cause identified as {root_cause}.")
        
    def calculate_failure_probability(self, component: str) -> float:
        model = self.failure_models.get(component, {"mtbf_days": 365})
        mtbf = model["mtbf_days"]
        
        probability = 1 - (0.5 ** (30 / mtbf))
        return round(probability, 2)
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            result = self.diagnose_fault(message.payload)
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=result
            )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "diagnose_fault":
            data = task.get("data", {})
            result = self.diagnose_fault(data)
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}


class CustomerEngagementAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="CUSTOMER_ENGAGEMENT_001",
            agent_name="Customer Engagement Agent",
            agent_type="worker"
        )
        
        self.customer_profiles = {}
        
        self.voice_templates = {
            "critical": "Attention. A critical issue has been detected in your vehicle. Fault code {dtc} indicates {issue}. For your safety, I have tentatively booked a service appointment for {slot}. Would you like to confirm this appointment, or choose a different time?",
            "high": "Hello. This is your vehicle's predictive maintenance system. We've detected a potential issue that requires attention. Fault code {dtc} suggests {issue}. I've scheduled a tentative service appointment for {slot}. Please confirm or let me know a better time.",
            "medium": "Good day. Your vehicle's diagnostic system has identified a maintenance need. {issue}. A service appointment has been tentatively scheduled for {slot}. Would you like to confirm?",
            "low": "Hi. A routine maintenance alert for your vehicle. {issue}. You may want to schedule a service visit at your convenience."
        }
        
        self.notification_templates = {
            "critical": "URGENT: Critical vehicle fault detected! {issue}. Service booked for {slot}. Please confirm.",
            "high": "Vehicle Alert: {issue}. Service scheduled for {slot}. Tap to confirm or reschedule.",
            "medium": "Maintenance Needed: {issue}. Tentative appointment: {slot}.",
            "low": "Reminder: {issue}. Consider scheduling service."
        }
        
    def generate_notification(self, diagnosis: Dict[str, Any], slot: Dict[str, Any] = None) -> Dict[str, Any]:
        
        severity = diagnosis.get("severity", "medium")
        dtc = diagnosis.get("dtc_code", "UNKNOWN")
        issue = diagnosis.get("root_cause", "potential issue")
        reasoning = diagnosis.get("reasoning", "")
        
        slot_str = "Tomorrow at 10:00 AM"
        if slot:
            slot_str = f"{slot.get('date', 'Tomorrow')} at {slot.get('time', '10:00 AM')}"
            
        voice_script = self.voice_templates[severity].format(
            dtc=dtc,
            issue=issue,
            slot=slot_str
        )
        
        notification_text = self.notification_templates[severity].format(
            dtc=dtc,
            issue=issue,
            slot=slot_str
        )
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "severity": severity,
            "dtc_code": dtc,
            "voice_script": voice_script,
            "notification_text": notification_text,
            "detailed_explanation": reasoning,
            "tentative_slot": slot_str,
            "action_required": severity in ["critical", "high"],
            "channels": ["voice", "mobile_app"]
        }
        
        self.log_action(
            action="generate_notification",
            details={
                "severity": severity,
                "dtc": dtc,
                "channels": result["channels"]
            }
        )
        
        return result
        
    def process_customer_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        
        action = response.get("action", "")
        
        result = {
            "timestamp": datetime.now().isoformat(),
            "customer_action": action,
            "next_step": ""
        }
        
        if action == "confirm":
            result["next_step"] = "confirm_booking"
            result["voice_response"] = "Excellent! Your appointment has been confirmed. You'll receive a confirmation message shortly with service center details."
            
        elif action == "reschedule":
            result["next_step"] = "show_calendar"
            result["voice_response"] = "No problem. Let me show you the available time slots. When would work best for you?"
            
        elif action == "deny":
            result["next_step"] = "log_denial"
            result["voice_response"] = "I understand. The tentative booking has been cancelled. Please note that the detected issue may require attention soon. You can schedule service anytime through the app."
            
        else:
            result["next_step"] = "await_response"
            result["voice_response"] = "I didn't catch that. Would you like to confirm the appointment, reschedule, or cancel?"
            
        self.log_action(
            action="process_customer_response",
            details={"action": action, "next_step": result["next_step"]}
        )
        
        return result
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            if "diagnosis" in message.payload:
                result = self.generate_notification(message.payload["diagnosis"])
            elif "customer_response" in message.payload:
                result = self.process_customer_response(message.payload["customer_response"])
            else:
                result = {"error": "Unknown payload type"}
                
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=result
            )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "notify_customer":
            data = task.get("data", {})
            result = self.generate_notification(data)
            self.status = AgentStatus.COMPLETED
            return result
            
        elif task_type == "process_response":
            response = task.get("response", {})
            result = self.process_customer_response(response)
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}


class SchedulingAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="SCHEDULING_001",
            agent_name="Scheduling Agent",
            agent_type="worker"
        )
        
        self.service_centers = {
            "SC001": {
                "name": "Downtown Service Hub",
                "address": "123 Main Street",
                "capacity": 10,
                "specialties": ["general", "electrical", "hybrid"]
            },
            "SC002": {
                "name": "Airport Road Center",
                "address": "456 Airport Rd",
                "capacity": 15,
                "specialties": ["general", "body", "transmission"]
            }
        }
        
        self.available_slots = self.generate_available_slots()
        
        self.appointments: Dict[str, Dict] = {}
        
    def generate_available_slots(self) -> List[Dict]:
        slots = []
        base_date = datetime.now() + timedelta(days=1)
        
        for day_offset in range(7):  # Next 7 days
            date = base_date + timedelta(days=day_offset)
            if date.weekday() < 6:  # Not Sunday
                for hour in [9, 10, 11, 14, 15, 16]:  # Business hours
                    slots.append({
                        "slot_id": f"SLOT_{date.strftime('%Y%m%d')}_{hour:02d}",
                        "date": date.strftime("%Y-%m-%d"),
                        "date_display": date.strftime("%A, %B %d"),
                        "time": f"{hour:02d}:00",
                        "service_center": "SC001",
                        "available": random.choice([True, True, True, False])  # 75% availability
                    })
        return slots
        
    def get_available_slots(self, urgency: str = "normal") -> List[Dict]:
        
        available = [s for s in self.available_slots if s["available"]]
        
        if urgency == "immediate":
            return available[:3]
        else:
            return available[:6]
            
    def book_tentative_slot(self, diagnosis: Dict[str, Any], customer_response: Dict[str, Any] = None) -> Dict[str, Any]:
        urgency = diagnosis.get("urgency", "schedule_soon")
        severity = diagnosis.get("severity", "medium")
        
        # Find best available slot
        slots = self.get_available_slots(urgency)
        
        if not slots:
            return {
                "success": False,
                "error": "No available slots",
                "next_available": "Please call service center directly"
            }
            
        selected_slot = slots[0]
        
        # Create appointment
        appointment_id = f"APT_{int(time.time())}"
        appointment = {
            "appointment_id": appointment_id,
            "status": "tentative",
            "slot": selected_slot,
            "diagnosis": diagnosis,
            "service_center": self.service_centers[selected_slot["service_center"]],
            "created_at": datetime.now().isoformat(),
            "severity": severity,
            "estimated_duration": "2 hours",
            "estimated_cost": diagnosis.get("estimated_repair_cost", 200)
        }
        
        self.appointments[appointment_id] = appointment
        
        result = {
            "success": True,
            "appointment_id": appointment_id,
            "status": "tentative",
            "date": selected_slot["date_display"],
            "time": selected_slot["time"],
            "service_center_name": appointment["service_center"]["name"],
            "service_center_address": appointment["service_center"]["address"],
            "estimated_duration": appointment["estimated_duration"],
            "estimated_cost": appointment["estimated_cost"],
            "alternative_slots": slots[1:4]  # Provide alternatives
        }
        
        self.log_action(
            action="book_tentative_slot",
            details={
                "appointment_id": appointment_id,
                "slot": selected_slot["slot_id"],
                "severity": severity
            }
        )
        
        return result
        
    def confirm_appointment(self, appointment_id: str) -> Dict[str, Any]:
        
        if appointment_id not in self.appointments:
            return {"success": False, "error": "Appointment not found"}
            
        appointment = self.appointments[appointment_id]
        appointment["status"] = "confirmed"
        appointment["confirmed_at"] = datetime.now().isoformat()
        
        # Mark slot as unavailable
        slot_id = appointment["slot"]["slot_id"]
        for slot in self.available_slots:
            if slot["slot_id"] == slot_id:
                slot["available"] = False
                break
                
        self.log_action(
            action="confirm_appointment",
            details={"appointment_id": appointment_id}
        )
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "status": "confirmed",
            "message": "Your appointment has been confirmed. See you soon!"
        }
        
    def reschedule_appointment(self, appointment_id: str, new_slot_id: str) -> Dict[str, Any]:
        
        if appointment_id not in self.appointments:
            return {"success": False, "error": "Appointment not found"}
            
        # Find new slot
        new_slot = None
        for slot in self.available_slots:
            if slot["slot_id"] == new_slot_id and slot["available"]:
                new_slot = slot
                break
                
        if not new_slot:
            return {"success": False, "error": "Requested slot not available"}
            
        appointment = self.appointments[appointment_id]
        old_slot = appointment["slot"]
        
        # Free old slot
        for slot in self.available_slots:
            if slot["slot_id"] == old_slot["slot_id"]:
                slot["available"] = True
                break
                
        # Book new slot
        appointment["slot"] = new_slot
        appointment["status"] = "rescheduled"
        appointment["rescheduled_at"] = datetime.now().isoformat()
        
        self.log_action(
            action="reschedule_appointment",
            details={
                "appointment_id": appointment_id,
                "old_slot": old_slot["slot_id"],
                "new_slot": new_slot_id
            }
        )
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "status": "rescheduled",
            "new_date": new_slot["date_display"],
            "new_time": new_slot["time"]
        }
        
    def cancel_appointment(self, appointment_id: str) -> Dict[str, Any]:
        
        if appointment_id not in self.appointments:
            return {"success": False, "error": "Appointment not found"}
            
        appointment = self.appointments[appointment_id]
        
        # Free the slot
        slot_id = appointment["slot"]["slot_id"]
        for slot in self.available_slots:
            if slot["slot_id"] == slot_id:
                slot["available"] = True
                break
                
        appointment["status"] = "cancelled"
        appointment["cancelled_at"] = datetime.now().isoformat()
        
        self.log_action(
            action="cancel_appointment",
            details={"appointment_id": appointment_id}
        )
        
        return {
            "success": True,
            "appointment_id": appointment_id,
            "status": "cancelled",
            "message": "Appointment cancelled. You can reschedule anytime."
        }
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            action = message.payload.get("action", "book")
            
            if action == "book":
                result = self.book_tentative_slot(message.payload.get("diagnosis", {}))
            elif action == "confirm":
                result = self.confirm_appointment(message.payload.get("appointment_id"))
            elif action == "reschedule":
                result = self.reschedule_appointment(
                    message.payload.get("appointment_id"),
                    message.payload.get("new_slot_id")
                )
            elif action == "cancel":
                result = self.cancel_appointment(message.payload.get("appointment_id"))
            elif action == "get_slots":
                result = {"slots": self.get_available_slots()}
            else:
                result = {"error": "Unknown action"}
                
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=result
            )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "book_tentative_slot":
            data = task.get("data", {})
            result = self.book_tentative_slot(data)
            self.status = AgentStatus.COMPLETED
            return result
            
        elif task_type == "confirm":
            result = self.confirm_appointment(task.get("appointment_id"))
            self.status = AgentStatus.COMPLETED
            return result
            
        elif task_type == "get_slots":
            result = {"slots": self.get_available_slots()}
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}


class FeedbackAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="FEEDBACK_001",
            agent_name="Feedback Agent",
            agent_type="worker"
        )
        
        self.feedback_records: List[Dict] = []
        
    def collect_feedback(self, appointment_id: str, feedback: Dict[str, Any]) -> Dict[str, Any]:
        
        record = {
            "feedback_id": f"FB_{int(time.time())}",
            "appointment_id": appointment_id,
            "timestamp": datetime.now().isoformat(),
            "rating": feedback.get("rating", 0),
            "comments": feedback.get("comments", ""),
            "service_quality": feedback.get("service_quality", 0),
            "would_recommend": feedback.get("would_recommend", False)
        }
        
        self.feedback_records.append(record)
        
        self.log_action(
            action="collect_feedback",
            details={
                "appointment_id": appointment_id,
                "rating": record["rating"]
            }
        )
        
        return {
            "success": True,
            "feedback_id": record["feedback_id"],
            "message": "Thank you for your feedback!"
        }
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            result = self.collect_feedback(
                message.payload.get("appointment_id"),
                message.payload.get("feedback", {})
            )
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=result
            )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "collect_feedback":
            result = self.collect_feedback(
                task.get("appointment_id"),
                task.get("feedback", {})
            )
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}


class ManufacturingInsightsAgent(BaseAgent):
    
    def __init__(self):
        super().__init__(
            agent_id="MANUFACTURING_001",
            agent_name="Manufacturing Insights Agent",
            agent_type="worker"
        )
        
        self.fault_records: List[Dict] = []
        self.pattern_analysis: Dict[str, int] = {}
        
    def log_for_capa(self, workflow_data: Dict[str, Any]) -> Dict[str, Any]:
        
        diagnosis = workflow_data.get("stages", {}).get("diagnosis", {})
        
        record = {
            "capa_id": f"CAPA_{int(time.time())}",
            "timestamp": datetime.now().isoformat(),
            "workflow_id": workflow_data.get("workflow_id"),
            "dtc_code": diagnosis.get("dtc_code"),
            "root_cause": diagnosis.get("root_cause"),
            "severity": diagnosis.get("severity"),
            "component": diagnosis.get("root_cause"),
            "failure_mode": diagnosis.get("reasoning"),
            "recommended_action": diagnosis.get("recommended_action")
        }
        
        self.fault_records.append(record)
        
        # Update pattern tracking
        component = record["component"] or "Unknown"
        self.pattern_analysis[component] = self.pattern_analysis.get(component, 0) + 1
        
        self.log_action(
            action="log_for_capa",
            details={
                "capa_id": record["capa_id"],
                "dtc_code": record["dtc_code"],
                "component": component
            }
        )
        
        return {
            "success": True,
            "capa_id": record["capa_id"],
            "pattern_detected": self.pattern_analysis[component] >= 3,
            "occurrence_count": self.pattern_analysis[component]
        }
        
    def get_quality_insights(self) -> Dict[str, Any]:
        
        return {
            "total_faults": len(self.fault_records),
            "component_breakdown": dict(self.pattern_analysis),
            "top_issues": sorted(
                self.pattern_analysis.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "recent_faults": self.fault_records[-10:] if self.fault_records else []
        }
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        if message.message_type == MessageType.DATA:
            if "workflow" in message.payload:
                result = self.log_for_capa(message.payload)
            else:
                result = self.get_quality_insights()
                
            return AgentMessage(
                sender=self.agent_id,
                receiver=message.sender,
                message_type=MessageType.RESPONSE,
                payload=result
            )
        return None
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "log_for_capa":
            data = task.get("data", {})
            result = self.log_for_capa(data)
            self.status = AgentStatus.COMPLETED
            return result
            
        elif task_type == "get_insights":
            result = self.get_quality_insights()
            self.status = AgentStatus.COMPLETED
            return result
            
        self.status = AgentStatus.ERROR
        return {"error": "Unknown task type"}
