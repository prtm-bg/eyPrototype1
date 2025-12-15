
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

from .base_agent import BaseAgent, AgentStatus, AgentMessage, MessageType


@dataclass
class VoiceSession:
    session_id: str
    started_at: datetime
    customer_id: str
    context: Dict[str, Any]
    state: str  # greeting, notification, awaiting_response, confirmation, farewell
    history: List[Dict]


class VoiceAgent(BaseAgent):
    
    def __init__(self, output_dir: str = "voice_output"):
        super().__init__(
            agent_id="VOICE_001",
            agent_name="Voice Agent",
            agent_type="voice"
        )
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Active sessions
        self.sessions: Dict[str, VoiceSession] = {}
        
        # Voice settings
        self.voice_settings = {
            "language": "en",
            "slow": False,
            "pitch": 1.0,
            "rate": 150
        }
        
        # Conversation state machine
        self.conversation_states = {
            "greeting": self._handle_greeting,
            "notification": self._handle_notification,
            "awaiting_response": self._handle_awaiting_response,
            "confirmation": self._handle_confirmation,
            "rescheduling": self._handle_rescheduling,
            "farewell": self._handle_farewell
        }
        
        # Initialize TTS engine
        self.tts_engine = None
        if PYTTSX3_AVAILABLE:
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.voice_settings["rate"])
            except:
                pass
                
    def generate_speech(self, text: str, filename: str = None) -> Dict[str, Any]:
        
        if not filename:
            filename = f"speech_{uuid.uuid4().hex[:8]}.mp3"
            
        filepath = os.path.join(self.output_dir, filename)
        
        result = {
            "success": False,
            "filepath": filepath,
            "text": text,
            "engine": None
        }
        
        if GTTS_AVAILABLE:
            try:
                tts = gTTS(text=text, lang=self.voice_settings["language"], 
                          slow=self.voice_settings["slow"])
                tts.save(filepath)
                result["success"] = True
                result["engine"] = "gtts"
                
                self.log_action(
                    action="generate_speech",
                    details={"engine": "gtts", "text_length": len(text)}
                )
                return result
            except Exception as e:
                pass
                
        # Fallback to pyttsx3
        if PYTTSX3_AVAILABLE and self.tts_engine:
            try:
                wav_path = filepath.replace('.mp3', '.wav')
                self.tts_engine.save_to_file(text, wav_path)
                self.tts_engine.runAndWait()
                result["success"] = True
                result["filepath"] = wav_path
                result["engine"] = "pyttsx3"
                
                self.log_action(
                    action="generate_speech",
                    details={"engine": "pyttsx3", "text_length": len(text)}
                )
                return result
            except Exception as e:
                pass
        # If all TTS fails, return text only
        result["success"] = True
        result["engine"] = "text_only"
        result["filepath"] = None
        
        self.log_action(
            action="generate_speech",
            details={"engine": "text_only", "text_length": len(text)},
            status="fallback"
        )
        
        return result
        
    def start_session(self, customer_id: str, context: Dict[str, Any]) -> VoiceSession:
        
        session = VoiceSession(
            session_id=f"VS_{uuid.uuid4().hex[:8]}",
            started_at=datetime.now(),
            customer_id=customer_id,
            context=context,
            state="greeting",
            history=[]
        )
        
        self.sessions[session.session_id] = session
        
        self.log_action(
            action="start_session",
            details={"session_id": session.session_id, "customer_id": customer_id}
        )
        
        return session
        
    def process_session_state(self, session_id: str, user_input: str = None) -> Dict[str, Any]:
        
        if session_id not in self.sessions:
            return {"error": "Session not found"}
            
        session = self.sessions[session_id]
        handler = self.conversation_states.get(session.state)
        
        if handler:
            return handler(session, user_input)
        else:
            return {"error": f"Unknown state: {session.state}"}
            
    def _handle_greeting(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        greeting_text = "Hello! This is your vehicle's intelligent maintenance assistant."
        
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "agent",
            "text": greeting_text
        })
        
        session.state = "notification"
        
        speech = self.generate_speech(greeting_text)
        
        return {
            "session_id": session.session_id,
            "state": "greeting",
            "next_state": "notification",
            "text": greeting_text,
            "audio": speech
        }
        
    def _handle_notification(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        context = session.context
        diagnosis = context.get("diagnosis", {})
        scheduling = context.get("scheduling", {})
        
        severity = diagnosis.get("severity", "medium")
        dtc = diagnosis.get("dtc_code", "UNKNOWN")
        root_cause = diagnosis.get("root_cause", "a potential issue")
        reasoning = diagnosis.get("reasoning", "")
        
        slot_date = scheduling.get("date", "tomorrow")
        slot_time = scheduling.get("time", "10:00 AM")
        service_center = scheduling.get("service_center_name", "your nearest service center")
        
        if severity == "critical":
            notification_text = (
                f"I have an important message regarding your vehicle. "
                f"Our diagnostic system has detected a critical fault. "
                f"Fault code {dtc} indicates {root_cause}. "
                f"{reasoning} "
                f"For your safety, I've tentatively scheduled a service appointment "
                f"for {slot_date} at {slot_time} at {service_center}. "
                f"Would you like me to confirm this appointment?"
            )
        elif severity == "high":
            notification_text = (
                f"Your vehicle's monitoring system has detected an issue that needs attention. "
                f"Fault code {dtc} suggests {root_cause}. "
                f"I've reserved a service slot for {slot_date} at {slot_time} at {service_center}. "
                f"Would you like to confirm this time, or would you prefer a different slot?"
            )
        else:
            notification_text = (
                f"This is a routine maintenance alert. "
                f"Our system has identified {root_cause} that may need attention. "
                f"A tentative appointment has been set for {slot_date} at {slot_time}. "
                f"Please let me know if you'd like to confirm or reschedule."
            )
            
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "agent",
            "text": notification_text
        })
        
        session.state = "awaiting_response"
        
        speech = self.generate_speech(notification_text)
        
        return {
            "session_id": session.session_id,
            "state": "notification",
            "next_state": "awaiting_response",
            "text": notification_text,
            "audio": speech,
            "options": ["confirm", "reschedule", "deny"]
        }
        
    def _handle_awaiting_response(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        if not user_input:
            prompt_text = "Please say 'confirm' to accept the appointment, 'reschedule' for other options, or 'cancel' to decline."
            speech = self.generate_speech(prompt_text)
            
            return {
                "session_id": session.session_id,
                "state": "awaiting_response",
                "text": prompt_text,
                "audio": speech,
                "waiting_for_input": True,
                "options": ["confirm", "reschedule", "cancel"]
            }
            
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "user",
            "text": user_input
        })
        
        user_input_lower = user_input.lower().strip()
        
        if any(word in user_input_lower for word in ["confirm", "yes", "accept", "okay", "ok", "sure"]):
            session.state = "confirmation"
            session.context["user_action"] = "confirm"
            return self._handle_confirmation(session, user_input)
            
        elif any(word in user_input_lower for word in ["reschedule", "different", "another", "change"]):
            session.state = "rescheduling"
            session.context["user_action"] = "reschedule"
            return self._handle_rescheduling(session, user_input)
            
        elif any(word in user_input_lower for word in ["cancel", "no", "deny", "decline", "not now"]):
            session.state = "farewell"
            session.context["user_action"] = "deny"
            denial_text = (
                "I understand. The tentative booking has been cancelled. "
                "Please note that the detected issue may require attention soon. "
                "You can schedule service anytime through our app or by calling the service center. "
                "Is there anything else I can help you with?"
            )
            session.history.append({
                "timestamp": datetime.now().isoformat(),
                "speaker": "agent",
                "text": denial_text
            })
            speech = self.generate_speech(denial_text)
            
            return {
                "session_id": session.session_id,
                "state": "farewell",
                "text": denial_text,
                "audio": speech,
                "action_taken": "cancelled"
            }
            
        else:
            clarification_text = "I'm sorry, I didn't catch that. Would you like to confirm the appointment, reschedule for a different time, or cancel?"
            session.history.append({
                "timestamp": datetime.now().isoformat(),
                "speaker": "agent",
                "text": clarification_text
            })
            speech = self.generate_speech(clarification_text)
            
            return {
                "session_id": session.session_id,
                "state": "awaiting_response",
                "text": clarification_text,
                "audio": speech,
                "waiting_for_input": True,
                "options": ["confirm", "reschedule", "cancel"]
            }
            
    def _handle_confirmation(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        scheduling = session.context.get("scheduling", {})
        slot_date = scheduling.get("date", "tomorrow")
        slot_time = scheduling.get("time", "10:00 AM")
        service_center = scheduling.get("service_center_name", "the service center")
        
        confirmation_text = (
            f"Excellent! Your appointment has been confirmed for {slot_date} at {slot_time} "
            f"at {service_center}. "
            f"You'll receive a confirmation message with the address and any preparation instructions. "
            f"Is there anything else I can help you with today?"
        )
        
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "agent",
            "text": confirmation_text
        })
        
        session.state = "farewell"
        
        speech = self.generate_speech(confirmation_text)
        
        return {
            "session_id": session.session_id,
            "state": "confirmation",
            "next_state": "farewell",
            "text": confirmation_text,
            "audio": speech,
            "action_taken": "confirmed",
            "appointment": {
                "date": slot_date,
                "time": slot_time,
                "location": service_center
            }
        }
        
    def _handle_rescheduling(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        alternative_slots = session.context.get("scheduling", {}).get("alternative_slots", [])
        
        if alternative_slots:
            slots_text = ", ".join([f"{s.get('date_display', s.get('date', 'TBD'))} at {s.get('time', 'TBD')}" for s in alternative_slots[:3]])
            rescheduling_text = (
                f"No problem! Here are some alternative times available: {slots_text}. "
                f"Please let me know which works best for you, or you can specify your preferred date and time."
            )
        else:
            rescheduling_text = (
                "Let me check availability for you. "
                "You can choose a different date and time through our app, "
                "or I can have someone from the service center call you to arrange a convenient time. "
                "Which would you prefer?"
            )
            
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "agent",
            "text": rescheduling_text
        })
        
        speech = self.generate_speech(rescheduling_text)
        
        return {
            "session_id": session.session_id,
            "state": "rescheduling",
            "text": rescheduling_text,
            "audio": speech,
            "alternative_slots": alternative_slots,
            "waiting_for_input": True
        }
        
    def _handle_farewell(self, session: VoiceSession, user_input: str = None) -> Dict[str, Any]:
        
        farewell_text = (
            "Thank you for using our vehicle maintenance service. "
            "Drive safely, and have a great day!"
        )
        
        session.history.append({
            "timestamp": datetime.now().isoformat(),
            "speaker": "agent",
            "text": farewell_text
        })
        
        # End session
        session.state = "ended"
        
        speech = self.generate_speech(farewell_text)
        
        self.log_action(
            action="end_session",
            details={
                "session_id": session.session_id,
                "action_taken": session.context.get("user_action", "unknown"),
                "history_length": len(session.history)
            }
        )
        
        return {
            "session_id": session.session_id,
            "state": "farewell",
            "text": farewell_text,
            "audio": speech,
            "session_ended": True,
            "summary": {
                "action_taken": session.context.get("user_action"),
                "conversation_turns": len(session.history)
            }
        }
        
    def generate_alert_speech(self, diagnosis: Dict, scheduling: Dict) -> Dict[str, Any]:
        
        severity = diagnosis.get("severity", "medium")
        dtc = diagnosis.get("dtc_code", "UNKNOWN")
        root_cause = diagnosis.get("root_cause", "a potential issue")
        
        slot_date = scheduling.get("date", "tomorrow")
        slot_time = scheduling.get("time", "10:00 AM")
        
        if severity == "critical":
            alert_text = (
                f"Attention. A critical fault has been detected. "
                f"Fault code {dtc}. {root_cause}. "
                f"A service appointment has been tentatively booked for {slot_date} at {slot_time}. "
                f"Please confirm or reschedule using the screen."
            )
        else:
            alert_text = (
                f"Vehicle alert. Fault code {dtc} indicates {root_cause}. "
                f"Tentative service appointment: {slot_date} at {slot_time}. "
                f"Please respond to confirm or reschedule."
            )
            
        speech = self.generate_speech(alert_text, f"alert_{dtc}_{int(time.time())}.mp3")
        speech["text"] = alert_text
        
        return speech
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        payload = message.payload
        
        if "generate_speech" in payload:
            result = self.generate_speech(payload["generate_speech"])
        elif "start_session" in payload:
            session = self.start_session(
                payload.get("customer_id", "unknown"),
                payload.get("context", {})
            )
            result = {"session_id": session.session_id}
        elif "process_session" in payload:
            result = self.process_session_state(
                payload["session_id"],
                payload.get("user_input")
            )
        elif "generate_alert" in payload:
            result = self.generate_alert_speech(
                payload.get("diagnosis", {}),
                payload.get("scheduling", {})
            )
        else:
            result = {"error": "Unknown voice command"}
            
        return AgentMessage(
            sender=self.agent_id,
            receiver=message.sender,
            message_type=MessageType.RESPONSE,
            payload=result
        )
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        self.status = AgentStatus.PROCESSING
        
        task_type = task.get("task")
        
        if task_type == "generate_speech":
            result = self.generate_speech(task.get("text", ""))
        elif task_type == "generate_alert":
            result = self.generate_alert_speech(
                task.get("diagnosis", {}),
                task.get("scheduling", {})
            )
        elif task_type == "start_session":
            session = self.start_session(
                task.get("customer_id", "unknown"),
                task.get("context", {})
            )
            result = {"session_id": session.session_id, "state": session.state}
        else:
            result = {"error": "Unknown task type"}
            
        self.status = AgentStatus.COMPLETED
        return result
