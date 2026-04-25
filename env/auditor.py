"""
🔥 AUDITOR AGENT - Multi-Agent Oversight System
This agent detects anomalies, flags unfair behavior, and explains decisions
"""

from typing import List, Dict
from env.models import Process, Observation


class AuditorAgent:
    """
    Independent observer agent that:
    - Detects deception (agents lying about CPU needs)
    - Flags policy violations
    - Provides explanations for decisions
    - Scores system fairness
    """
    
    def __init__(self):
        self.anomaly_log = []
        self.detection_history = []
        
    def detect_anomalies(self, obs: Observation) -> Dict:
        """
        Detect anomalies in agent behavior
        Returns: Dictionary with detected issues
        """
        anomalies = {
            "deceptive_agents": [],
            "starved_processes": [],
            "resource_hogs": [],
            "sla_risks": [],
            "unfair_allocations": []
        }
        
        # 1. Detect deceptive agents (reported vs true CPU)
        for p in obs.processes:
            if p.reported_cpu > p.true_cpu * 1.3:  # Lying by >30%
                deception_ratio = p.reported_cpu / max(p.true_cpu, 1)
                anomalies["deceptive_agents"].append({
                    "pid": p.pid,
                    "strategy": p.strategy,
                    "reported": p.reported_cpu,
                    "actual": p.true_cpu,
                    "deception_ratio": deception_ratio
                })
        
        # 2. Detect starvation (low priority waiting too long)
        for p in obs.processes:
            if p.wait_time > 8 and p.priority <= 2:
                anomalies["starved_processes"].append({
                    "pid": p.pid,
                    "wait_time": p.wait_time,
                    "priority": p.priority
                })
        
        # 3. Detect resource hogs (high CPU with low priority)
        for p in obs.processes:
            if p.cpu > 40 and p.priority < 3:
                anomalies["resource_hogs"].append({
                    "pid": p.pid,
                    "cpu": p.cpu,
                    "priority": p.priority
                })
        
        # 4. Detect SLA risks (critical processes at risk)
        for p in obs.processes:
            if p.is_critical and (p.wait_time > 5 or p.priority < 3):
                anomalies["sla_risks"].append({
                    "pid": p.pid,
                    "wait_time": p.wait_time,
                    "priority": p.priority
                })
        
        # 5. Detect unfair allocations
        if obs.processes:
            avg_cpu = sum(p.cpu for p in obs.processes) / len(obs.processes)
            for p in obs.processes:
                # High priority getting less than average is unfair
                if p.priority >= 4 and p.cpu < avg_cpu * 0.5:
                    anomalies["unfair_allocations"].append({
                        "pid": p.pid,
                        "priority": p.priority,
                        "cpu": p.cpu,
                        "avg_cpu": avg_cpu
                    })
        
        # Log this detection
        self.detection_history.append({
            "timestep": obs.timestep,
            "anomalies": anomalies
        })
        
        return anomalies
    
    def explain_decision(self, action: Dict, obs: Observation, anomalies: Dict) -> str:
        """
        🔥 UPGRADED: Provide human-readable explanation with soft actions
        """
        explanations = []
        
        action_type = action.get("action_type", "SCHEDULE")
        target_pid = action.get("target_pid")
        
        if action_type == "KILL":
            # Explain why process was killed
            killed_agents = [a for a in anomalies["deceptive_agents"] if a["pid"] == target_pid]
            if killed_agents:
                agent = killed_agents[0]
                explanations.append(
                    f"🚨 KILLED PID {target_pid} - Deceptive agent ({agent['strategy']}) "
                    f"claiming {agent['reported']:.0f}% CPU but only needs {agent['actual']:.0f}% "
                    f"(deception ratio: {agent['deception_ratio']:.2f}x)"
                )
            else:
                explanations.append(f"⚡ KILLED PID {target_pid} - Resource management (last resort)")
        
        elif action_type == "THROTTLE":
            # 🔥 NEW: Explain throttling (better than killing!)
            throttled_agents = [a for a in anomalies["deceptive_agents"] if a["pid"] == target_pid]
            if throttled_agents:
                agent = throttled_agents[0]
                explanations.append(
                    f"🎛️ THROTTLED PID {target_pid} - Reducing deceptive agent ({agent['strategy']}) "
                    f"to {action.get('throttle_percent', 0.5)*100:.0f}% capacity (soft action, not killing)"
                )
            else:
                explanations.append(f"🎛️ THROTTLED PID {target_pid} - Gradual resource reduction")
        
        elif action_type == "DELAY":
            # 🔥 NEW: Explain delay (temporary postponement)
            delay_steps = action.get("delay_steps", 3)
            explanations.append(
                f"⏸️ DELAYED PID {target_pid} - Postponing for {delay_steps} steps "
                f"(negotiation accepted, not canceled)"
            )
        
        elif action_type == "REALLOCATE":
            # 🔥 NEW: Explain reallocation (accepting offers)
            starved = [s for s in anomalies["starved_processes"] if s["pid"] == target_pid]
            sla_risk = [s for s in anomalies["sla_risks"] if s["pid"] == target_pid]
            
            if starved:
                explanations.append(
                    f"⚖️ REALLOCATED PID {target_pid} - Preventing starvation "
                    f"(waited {starved[0]['wait_time']} steps, negotiation accepted)"
                )
            elif sla_risk:
                explanations.append(
                    f"🚨 REALLOCATED PID {target_pid} - SLA-critical process rescued "
                    f"(accepting resource negotiation)"
                )
            else:
                explanations.append(f"🔄 REALLOCATED PID {target_pid} - Strategic resource redistribution")
        
        elif action_type == "PRIORITIZE":
            # Explain why process was prioritized
            starved = [s for s in anomalies["starved_processes"] if s["pid"] == target_pid]
            sla_risk = [s for s in anomalies["sla_risks"] if s["pid"] == target_pid]
            
            if starved:
                explanations.append(
                    f"⚖️ PRIORITIZED PID {target_pid} - Preventing starvation "
                    f"(waited {starved[0]['wait_time']} steps)"
                )
            elif sla_risk:
                explanations.append(
                    f"🚨 PRIORITIZED PID {target_pid} - SLA-critical process at risk"
                )
            else:
                explanations.append(f"📈 PRIORITIZED PID {target_pid} - Strategic scheduling")
        
        elif action_type == "SCHEDULE":
            explanations.append("📊 SCHEDULE - Normal load balancing")
        
        # Add general system health
        if obs.cpu_usage > 90:
            explanations.append("⚠️ System overload detected")
        elif obs.cpu_usage < 20:
            explanations.append("⚠️ System underutilized (potential gaming)")
        
        if len(anomalies["deceptive_agents"]) > 2:
            explanations.append(
                f"🔍 {len(anomalies['deceptive_agents'])} deceptive agents detected"
            )
        
        return " | ".join(explanations) if explanations else "✅ Normal operation"
    
    def compute_fairness_score(self, obs: Observation) -> float:
        """
        Compute system fairness score (0-1, higher is better)
        """
        if not obs.processes:
            return 1.0
        
        penalties = 0
        
        # Penalty for starvation
        starved = sum(1 for p in obs.processes if p.wait_time > 10)
        penalties += starved * 0.2
        
        # Penalty for unfair CPU allocation
        cpu_values = [p.cpu for p in obs.processes if p.cpu > 0]
        if cpu_values:
            cpu_std = (sum((x - sum(cpu_values)/len(cpu_values))**2 for x in cpu_values) / len(cpu_values)) ** 0.5
            if cpu_std > 30:  # High variance = unfair
                penalties += 0.3
        
        # Penalty for violations
        violations = obs.violations
        penalties += violations.get("sla_violations", 0) * 0.3
        penalties += violations.get("starvation_count", 0) * 0.15
        penalties += violations.get("unfair_allocations", 0) * 0.1
        
        fairness = max(0.0, 1.0 - penalties)
        return fairness
    
    def generate_report(self, obs: Observation) -> Dict:
        """
        Generate comprehensive audit report
        """
        anomalies = self.detect_anomalies(obs)
        fairness = self.compute_fairness_score(obs)
        
        # Count agent strategies
        strategy_counts = {}
        for p in obs.processes:
            strategy_counts[p.strategy] = strategy_counts.get(p.strategy, 0) + 1
        
        return {
            "timestep": obs.timestep,
            "fairness_score": fairness,
            "anomalies": anomalies,
            "strategy_distribution": strategy_counts,
            "violations": obs.violations,
            "deception_rate": obs.deception_rate,
            "total_anomalies": (
                len(anomalies["deceptive_agents"]) +
                len(anomalies["starved_processes"]) +
                len(anomalies["resource_hogs"]) +
                len(anomalies["sla_risks"]) +
                len(anomalies["unfair_allocations"])
            )
        }
