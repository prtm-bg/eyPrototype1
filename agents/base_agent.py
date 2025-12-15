
import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
from queue import Queue


class AgentStatus(Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


class MessageType(Enum):
    COMMAND = "command"
    DATA = "data"
    RESPONSE = "response"
    ALERT = "alert"
    STATUS = "status"
    HANDOFF = "handoff"


@dataclass
class AgentMessage:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    sender: str = ""
    receiver: str = ""
    message_type: MessageType = MessageType.DATA
    priority: int = 1  # 1=Low, 2=Medium, 3=High, 4=Critical
    payload: Dict[str, Any] = field(default_factory=dict)
    requires_response: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "timestamp": self.timestamp.isoformat(),
            "sender": self.sender,
            "receiver": self.receiver,
            "message_type": self.message_type.value,
            "priority": self.priority,
            "payload": self.payload,
            "requires_response": self.requires_response
        }


@dataclass
class AgentLog:
    timestamp: datetime
    agent_id: str
    action: str
    details: Dict[str, Any]
    status: str
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent_id": self.agent_id,
            "action": self.action,
            "details": self.details,
            "status": self.status
        }


class BaseAgent:
    def __init__(self, agent_id: str, agent_name: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.status = AgentStatus.IDLE
        self.message_queue: Queue = Queue()
        self.response_queue: Queue = Queue()
        self.action_log: List[AgentLog] = []
        self.master_agent = None
        self._running = False
        self.callbacks = {}
        
    def set_master(self, master_agent):
        self.master_agent = master_agent
        
    def log_action(self, action: str, details: Dict[str, Any], status: str = "success"):
        log_entry = AgentLog(
            timestamp=datetime.now(),
            agent_id=self.agent_id,
            action=action,
            details=details,
            status=status
        )
        self.action_log.append(log_entry)
        
        # Send to master for UEBA if registered
        if self.master_agent:
            self.master_agent.receive_log(log_entry)
            
        return log_entry
        
    def send_message(self, receiver: str, message_type: MessageType, 
                     payload: Dict[str, Any], priority: int = 2,
                     requires_response: bool = False) -> AgentMessage:
        message = AgentMessage(
            sender=self.agent_id,
            receiver=receiver,
            message_type=message_type,
            priority=priority,
            payload=payload,
            requires_response=requires_response
        )
        
        self.log_action(
            action=f"send_message_to_{receiver}",
            details={"message_type": message_type.value, "priority": priority}
        )
        
        if self.master_agent:
            return self.master_agent.route_message(message)
        return message
        
    def receive_message(self, message: AgentMessage):
        self.message_queue.put(message)
        self.log_action(
            action="receive_message",
            details={"from": message.sender, "type": message.message_type.value}
        )
        
    def process_message(self, message: AgentMessage) -> Optional[AgentMessage]:
        raise NotImplementedError("Subclasses must implement process_message")
        
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("Subclasses must implement execute_task")
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "agent_type": self.agent_type,
            "status": self.status.value,
            "queue_size": self.message_queue.qsize(),
            "log_count": len(self.action_log)
        }
        
    def register_callback(self, event_name: str, callback):
        self.callbacks[event_name] = callback
        
    def trigger_callback(self, event_name: str, data: Any = None):
        if event_name in self.callbacks:
            self.callbacks[event_name](data)
