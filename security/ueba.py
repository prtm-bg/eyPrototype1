
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class BehaviorBaseline:
    agent_id: str
    avg_actions_per_minute: float = 0.0
    common_actions: Dict[str, int] = field(default_factory=dict)
    common_receivers: Dict[str, int] = field(default_factory=dict)
    peak_hours: List[int] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class SecurityAlert:
    alert_id: str
    timestamp: datetime
    agent_id: str
    alert_type: str
    severity: str  # low, medium, high, critical
    description: str
    details: Dict[str, Any]
    action_taken: str
    resolved: bool = False


class UEBASecurityLayer:
    
    def __init__(self):
        self.baselines: Dict[str, BehaviorBaseline] = {}
        
        self.action_history: List[Dict] = []
        self.alerts: List[SecurityAlert] = []
        
        self.thresholds = {
            "actions_per_minute_multiplier": 3.0,  # Alert if actions > 3x baseline
            "unusual_action_threshold": 0.1,  # Alert if action is <10% of baseline
            "unauthorized_receivers": ["EXTERNAL", "UNKNOWN", "ADMIN"],
            "high_risk_actions": ["delete", "modify_config", "export_data", "escalate_privileges"],
            "min_baseline_actions": 50  # Minimum actions before baseline is reliable
        }
        
        self.security_rules = [
            self._rule_rapid_actions,
            self._rule_unusual_action,
            self._rule_unauthorized_receiver,
            self._rule_time_anomaly,
            self._rule_high_risk_action
        ]
        
        self.blocked_agents: Dict[str, datetime] = {}
        
    def log_action(self, agent_id: str, action: str, details: Dict[str, Any]) -> Optional[SecurityAlert]:
        
        timestamp = datetime.now()
        
        action_record = {
            "timestamp": timestamp,
            "agent_id": agent_id,
            "action": action,
            "details": details,
            "receiver": details.get("receiver", ""),
            "priority": details.get("priority", 1)
        }
        
        self.action_history.append(action_record)
        
        self._update_baseline(agent_id, action_record)
        
        if agent_id in self.blocked_agents:
            block_time = self.blocked_agents[agent_id]
            if (timestamp - block_time).seconds < 300:  # 5 minute block
                return self._create_alert(
                    agent_id, "blocked_agent_action", "high",
                    f"Blocked agent {agent_id} attempted action",
                    {"action": action, "blocked_at": block_time.isoformat()}
                )
            else:
                del self.blocked_agents[agent_id]
                
        for rule in self.security_rules:
            alert = rule(action_record)
            if alert:
                return alert
                
        return None
        
    def _update_baseline(self, agent_id: str, action_record: Dict):
        
        if agent_id not in self.baselines:
            self.baselines[agent_id] = BehaviorBaseline(agent_id=agent_id)
            
        baseline = self.baselines[agent_id]
        
        action = action_record["action"]
        baseline.common_actions[action] = baseline.common_actions.get(action, 0) + 1
        
        receiver = action_record.get("receiver", "")
        if receiver:
            baseline.common_receivers[receiver] = baseline.common_receivers.get(receiver, 0) + 1
            
        hour = action_record["timestamp"].hour
        if hour not in baseline.peak_hours:
            baseline.peak_hours.append(hour)
            
        recent_actions = [
            a for a in self.action_history[-1000:]
            if a["agent_id"] == agent_id and
            (datetime.now() - a["timestamp"]).seconds < 60
        ]
        baseline.avg_actions_per_minute = len(recent_actions)
        baseline.last_updated = datetime.now()
        
    def _rule_rapid_actions(self, action_record: Dict) -> Optional[SecurityAlert]:
        
        agent_id = action_record["agent_id"]
        
        if agent_id not in self.baselines:
            return None
            
        baseline = self.baselines[agent_id]
        total_actions = sum(baseline.common_actions.values())
        
        if total_actions < self.thresholds["min_baseline_actions"]:
            return None
            
        recent_actions = [
            a for a in self.action_history[-100:]
            if a["agent_id"] == agent_id and
            (datetime.now() - a["timestamp"]).seconds < 60
        ]
        
        expected_rate = total_actions / max(1, (datetime.now() - baseline.last_updated).seconds / 60)
        current_rate = len(recent_actions)
        
        if current_rate > expected_rate * self.thresholds["actions_per_minute_multiplier"]:
            return self._create_alert(
                agent_id, "rapid_actions", "medium",
                f"Agent {agent_id} performing actions {current_rate/max(0.1, expected_rate):.1f}x faster than baseline",
                {"current_rate": current_rate, "expected_rate": expected_rate}
            )
            
        return None
        
    def _rule_unusual_action(self, action_record: Dict) -> Optional[SecurityAlert]:
        
        agent_id = action_record["agent_id"]
        action = action_record["action"]
        
        if agent_id not in self.baselines:
            return None
            
        baseline = self.baselines[agent_id]
        total_actions = sum(baseline.common_actions.values())
        
        if total_actions < self.thresholds["min_baseline_actions"]:
            return None
            
        action_count = baseline.common_actions.get(action, 0)
        action_ratio = action_count / total_actions
        
        if action_ratio < self.thresholds["unusual_action_threshold"] and action_count < 5:
            return self._create_alert(
                agent_id, "unusual_action", "low",
                f"Agent {agent_id} performed unusual action: {action}",
                {"action": action, "historical_ratio": action_ratio}
            )
            
        return None
        
    def _rule_unauthorized_receiver(self, action_record: Dict) -> Optional[SecurityAlert]:
        
        receiver = action_record.get("receiver", "")
        agent_id = action_record["agent_id"]
        
        if receiver in self.thresholds["unauthorized_receivers"]:
            alert = self._create_alert(
                agent_id, "unauthorized_receiver", "high",
                f"Agent {agent_id} attempted to communicate with unauthorized receiver: {receiver}",
                {"receiver": receiver, "action": action_record["action"]}
            )
            
            self.blocked_agents[agent_id] = datetime.now()
            alert.action_taken = "agent_blocked"
            
            return alert
            
        return None
        
    def _rule_time_anomaly(self, action_record: Dict) -> Optional[SecurityAlert]:
        
        agent_id = action_record["agent_id"]
        hour = action_record["timestamp"].hour
        
        if agent_id not in self.baselines:
            return None
            
        baseline = self.baselines[agent_id]
        
        if baseline.peak_hours and hour not in baseline.peak_hours:
            if len(baseline.peak_hours) >= 5:
                return self._create_alert(
                    agent_id, "time_anomaly", "low",
                    f"Agent {agent_id} active at unusual hour: {hour}:00",
                    {"hour": hour, "peak_hours": baseline.peak_hours}
                )
                
        return None
        
    def _rule_high_risk_action(self, action_record: Dict) -> Optional[SecurityAlert]:
        
        action = action_record["action"]
        agent_id = action_record["agent_id"]
        
        for risk_action in self.thresholds["high_risk_actions"]:
            if risk_action in action.lower():
                return self._create_alert(
                    agent_id, "high_risk_action", "critical",
                    f"Agent {agent_id} performed high-risk action: {action}",
                    {"action": action, "details": action_record.get("details", {})}
                )
                
        return None
        
    def _create_alert(self, agent_id: str, alert_type: str, severity: str,
                      description: str, details: Dict) -> SecurityAlert:
        
        alert = SecurityAlert(
            alert_id=f"ALERT_{len(self.alerts)+1}_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            agent_id=agent_id,
            alert_type=alert_type,
            severity=severity,
            description=description,
            details=details,
            action_taken="logged"
        )
        
        self.alerts.append(alert)
        
        return alert
        
    def get_security_summary(self) -> Dict[str, Any]:
        
        now = datetime.now()
        recent_alerts = [
            a for a in self.alerts
            if (now - a.timestamp).seconds < 3600
        ]
        
        alert_counts = defaultdict(int)
        for alert in recent_alerts:
            alert_counts[alert.severity] += 1
            
        return {
            "total_alerts": len(self.alerts),
            "recent_alerts_1h": len(recent_alerts),
            "alert_breakdown": dict(alert_counts),
            "blocked_agents": list(self.blocked_agents.keys()),
            "agents_monitored": len(self.baselines),
            "total_actions_logged": len(self.action_history)
        }
        
    def get_agent_profile(self, agent_id: str) -> Dict[str, Any]:
        
        if agent_id not in self.baselines:
            return {"error": "Agent not found"}
            
        baseline = self.baselines[agent_id]
        
        agent_alerts = [
            {
                "alert_id": a.alert_id,
                "type": a.alert_type,
                "severity": a.severity,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alerts
            if a.agent_id == agent_id
        ][-10:]  
        
        return {
            "agent_id": agent_id,
            "total_actions": sum(baseline.common_actions.values()),
            "action_types": len(baseline.common_actions),
            "top_actions": sorted(
                baseline.common_actions.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "peak_hours": baseline.peak_hours,
            "avg_actions_per_minute": round(baseline.avg_actions_per_minute, 2),
            "is_blocked": agent_id in self.blocked_agents,
            "recent_alerts": agent_alerts
        }
        
    def get_recent_alerts(self, count: int = 10, severity: str = None) -> List[Dict]:
        
        alerts = self.alerts
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
            
        return [
            {
                "alert_id": a.alert_id,
                "timestamp": a.timestamp.isoformat(),
                "agent_id": a.agent_id,
                "type": a.alert_type,
                "severity": a.severity,
                "description": a.description,
                "action_taken": a.action_taken
            }
            for a in sorted(alerts, key=lambda x: x.timestamp, reverse=True)[:count]
        ]
        
    def resolve_alert(self, alert_id: str) -> bool:
        
        for alert in self.alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                return True
                
        return False
        
    def unblock_agent(self, agent_id: str) -> bool:
        
        if agent_id in self.blocked_agents:
            del self.blocked_agents[agent_id]
            return True
            
        return False
