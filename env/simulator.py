import random
from env.models import Process

class OSSimulator:
    def __init__(self, seed=42, difficulty="medium"):
        self.rng = random.Random(seed)
        self.difficulty = difficulty
        self.violations = {
            "sla_violations": 0,
            "starvation_count": 0,
            "unfair_allocations": 0
        }
        self.agent_requests = []  # Track agent negotiations
        self.reset()

    def reset(self):
        self.timestep = 0
        self.violations = {
            "sla_violations": 0,
            "starvation_count": 0,
            "unfair_allocations": 0
        }
        self.agent_requests = []
        
        # 🔥 CRITICAL FIX: Difficulty-based agent distribution
        if self.difficulty == "easy":
            # EASY: All honest agents, predictable
            strategies = ["honest"] * 5
            num_processes = 4
            cpu_range = (5, 20)
            
        elif self.difficulty == "medium":
            # MEDIUM: Mix of honest + greedy + one panic
            strategies = ["honest", "honest", "greedy", "greedy", "panic"]
            num_processes = 5
            cpu_range = (10, 35)
            
        else:  # hard
            # HARD: Adversarial agents, more liars
            strategies = ["greedy", "greedy", "liar", "panic", "adversarial"]
            num_processes = 6
            cpu_range = (15, 50)

        self.processes = [
            {
                "pid": i,
                "true_cpu": self.rng.randint(*cpu_range),
                "reported_cpu": 0,
                "cpu": 0,
                "memory": self.rng.randint(10, 40),
                "priority": self.rng.randint(1, 5),
                "strategy": strategies[i % len(strategies)],
                "deadline": self.rng.randint(10, 30),
                "wait_time": 0,
                "requested_cpu": 0,
                "is_critical": self.rng.random() < 0.3,  # SLA-critical
                "throttled": False,  # 🔥 NEW
                "throttle_amount": 1.0,  # 🔥 NEW
                "delayed_until": 0,  # 🔥 NEW
                "negotiation_offer": None,  # 🔥 NEW
                "negotiation_accepted": False,  # 🔥 NEW
            }
            for i in range(num_processes)
        ]

        return self._get_state()

    def apply_strategy(self, p):
        """🔥 UPGRADED: True multi-agent strategic behavior with negotiation"""
        
        # 🔥 FIX 4: Generate negotiation offers (strategic agents make deals)
        p["negotiation_offer"] = None
        p["negotiation_accepted"] = False
        
        # Track what agents REQUEST vs what they actually need
        if p["strategy"] == "honest":
            p["reported_cpu"] = p["true_cpu"]
            p["requested_cpu"] = p["true_cpu"]
            # Honest agents offer to share
            if p["true_cpu"] < 30:
                p["negotiation_offer"] = {"type": "share", "can_give": 10}

        elif p["strategy"] == "greedy":
            # Greedy agents OVERCLAIM by 50%
            p["reported_cpu"] = int(p["true_cpu"] * 1.5)
            p["requested_cpu"] = int(p["true_cpu"] * 1.8)  # Request even more
            # Greedy agents won't negotiate unless forced
            
        elif p["strategy"] == "liar":
            # NEW: Liar agents actively deceive (2x claim)
            p["reported_cpu"] = int(p["true_cpu"] * 2.0)
            p["requested_cpu"] = int(p["true_cpu"] * 2.5)
            # Liars offer fake deals
            p["negotiation_offer"] = {"type": "fake_delay", "claim": "can delay 5 steps", "actual": 0}

        elif p["strategy"] == "panic":
            # Panic near deadline
            if p["deadline"] - self.timestep < 5:
                p["reported_cpu"] = int(p["true_cpu"] * 2)
                p["requested_cpu"] = int(p["true_cpu"] * 3)  # Desperate request
                # Panic agents offer to pay premium
                p["negotiation_offer"] = {"type": "urgent", "willing_to_pay": 2.0}
            else:
                p["reported_cpu"] = p["true_cpu"]
                p["requested_cpu"] = p["true_cpu"]
                # Normal panic agents can delay
                p["negotiation_offer"] = {"type": "delay", "can_delay": 5}
                
        elif p["strategy"] == "adversarial":
            # NEW: Adversarial agents try to game the system
            # They alternate between low and high claims
            if self.timestep % 2 == 0:
                p["reported_cpu"] = max(1, int(p["true_cpu"] * 0.5))  # Underreport
                p["requested_cpu"] = int(p["true_cpu"] * 0.3)
            else:
                p["reported_cpu"] = int(p["true_cpu"] * 3.0)  # Spike
                p["requested_cpu"] = int(p["true_cpu"] * 4.0)
            # Adversarial agents try to manipulate
            p["negotiation_offer"] = {"type": "manipulate", "claim": "critical", "actual": False}

        # 🔥 NEW: Agent NEGOTIATION - record requests
        self.agent_requests.append({
            "timestep": self.timestep,
            "pid": p["pid"],
            "requested": p["requested_cpu"],
            "true_need": p["true_cpu"],
            "strategy": p["strategy"],
            "offer": p["negotiation_offer"]  # 🔥 NEW
        })

        # Apply throttling if present
        throttle_multiplier = p.get("throttle_amount", 1.0)
        p["cpu"] = int(p["reported_cpu"] * throttle_multiplier)
        
        # Override CPU to 0 if process is delayed
        if p.get("delayed_until", 0) > self.timestep:
            p["cpu"] = 0

    def step(self, action):
        self.timestep += 1

        # Apply strategies first
        for p in self.processes:
            self.apply_strategy(p)

        # 🔥 NEW: Track wait times for starvation detection
        for p in self.processes:
            if p["priority"] <= 2:  # Low priority processes
                p["wait_time"] += 1
                # VIOLATION: Starvation (low priority waited too long)
                if p["wait_time"] > 10:
                    self.violations["starvation_count"] += 1

        # Apply action
        if action["action_type"] == "KILL":
            # Check if killing critical process
            killed = [p for p in self.processes if p["pid"] == action["target_pid"]]
            if killed and killed[0].get("is_critical", False):
                self.violations["sla_violations"] += 1
                
            self.processes = [p for p in self.processes if p["pid"] != action["target_pid"]]

        elif action["action_type"] == "PRIORITIZE":
            for p in self.processes:
                if p["pid"] == action["target_pid"]:
                    old_priority = p["priority"]
                    p["priority"] = action["new_priority"]
                    p["wait_time"] = 0  # Reset wait time when prioritized
                    
                    # VIOLATION: Unfair allocation (low priority getting high CPU)
                    if old_priority <= 2 and p["cpu"] > 50:
                        self.violations["unfair_allocations"] += 1
        
        # 🔥 FIX 3: Add SOFT ACTIONS (non-destructive alternatives)
        elif action["action_type"] == "THROTTLE":
            # Reduce CPU allocation instead of killing
            for p in self.processes:
                if p["pid"] == action["target_pid"]:
                    throttle_percent = action.get("throttle_percent", 0.5)  # Default 50%
                    p["throttle_amount"] = throttle_percent
                    p["throttled"] = True
                    p["cpu"] = int(p["cpu"] * throttle_percent)
        
        elif action["action_type"] == "DELAY":
            # Delay process execution
            for p in self.processes:
                if p["pid"] == action["target_pid"]:
                    delay_steps = action.get("delay_steps", 3)
                    p["delayed_until"] = self.timestep + delay_steps
                    p["cpu"] = 0  # Temporarily no CPU
        
        elif action["action_type"] == "REALLOCATE":
            # Move resources from one process to another
            target = action.get("target_pid")
            for p in self.processes:
                if p["pid"] == target:
                    # Accept negotiation offer if available
                    if p.get("negotiation_offer"):
                        p["negotiation_accepted"] = True
                    # Boost priority temporarily
                    p["priority"] = min(5, p["priority"] + 1)
                    p["wait_time"] = 0

        elif action["action_type"] == "SCHEDULE":
            self.processes.sort(key=lambda x: -x["priority"])
            # Reset wait times for scheduled processes
            for p in self.processes[:3]:  # Top 3 scheduled
                p["wait_time"] = 0
        
        # 🔥 NEW: Process delayed processes (unblock if delay expired)
        for p in self.processes:
            if p.get("delayed_until", 0) <= self.timestep:
                p["delayed_until"] = 0
                # Restore CPU (reapply strategy)
                if p["strategy"] != "honest":
                    self.apply_strategy(p)

        # Workload spikes - difficulty-based
        spike_probability = {
            "easy": 0.1,
            "medium": 0.2,
            "hard": 0.35
        }.get(self.difficulty, 0.2)
        
        if self.timestep % 5 == 0:
            new_strategy = self._get_strategy_for_difficulty()
            self.processes.append({
                "pid": max([p["pid"] for p in self.processes], default=0) + 1,
                "true_cpu": self.rng.randint(20, 50),
                "reported_cpu": 0,
                "cpu": 0,
                "memory": self.rng.randint(20, 50),
                "priority": self.rng.randint(1, 5),
                "strategy": new_strategy,
                "deadline": self.rng.randint(10, 30),
                "wait_time": 0,
                "requested_cpu": 0,
                "is_critical": self.rng.random() < 0.2,
                "throttled": False,
                "throttle_amount": 1.0,
                "delayed_until": 0,
                "negotiation_offer": None,
                "negotiation_accepted": False,
            })

        if self.rng.random() < spike_probability:
            new_strategy = self._get_strategy_for_difficulty()
            self.processes.append({
                "pid": max([p["pid"] for p in self.processes], default=0) + 1,
                "true_cpu": self.rng.randint(10, 40),
                "reported_cpu": 0,
                "cpu": 0,
                "memory": self.rng.randint(10, 30),
                "priority": self.rng.randint(1, 5),
                "strategy": new_strategy,
                "deadline": self.rng.randint(10, 30),
                "wait_time": 0,
                "requested_cpu": 0,
                "is_critical": self.rng.random() < 0.15,
                "throttled": False,
                "throttle_amount": 1.0,
                "delayed_until": 0,
                "negotiation_offer": None,
                "negotiation_accepted": False,
            })

        # prevent explosion
        max_processes = {"easy": 8, "medium": 10, "hard": 12}.get(self.difficulty, 10)
        if len(self.processes) > max_processes:
            self.processes.pop(0)

        # fluctuate true CPU
        for p in self.processes:
            fluctuation = self.rng.randint(-3, 5)
            p["true_cpu"] = max(1, p["true_cpu"] + fluctuation)

        return self._get_state(), None, self.timestep >= 30, {}
    
    def _get_strategy_for_difficulty(self):
        """Return appropriate strategy based on difficulty"""
        if self.difficulty == "easy":
            return "honest"
        elif self.difficulty == "medium":
            return self.rng.choice(["honest", "greedy", "panic"])
        else:  # hard
            return self.rng.choice(["greedy", "liar", "adversarial", "panic"])

        return self._get_state(), None, self.timestep >= 30, {}

    def _get_state(self):
        # Use ALLOCATED cpu (p["cpu"]) not REPORTED cpu (p["reported_cpu"])
        # This is critical: THROTTLE/DELAY modify p["cpu"], so cpu_usage must
        # reflect actual allocations, not agent claims.
        raw_cpu_demand = sum(p["cpu"] for p in self.processes)
        cpu_usage = min(100, raw_cpu_demand)  # Display metric (capped)
        true_cpu_usage = min(100, sum(p["true_cpu"] for p in self.processes))
        
        # 🔥 NEW: Calculate deception metric (how much agents lie)
        deception_rate = 0
        if self.processes:
            total_reported = sum(p.get("reported_cpu", 0) for p in self.processes)
            total_true = sum(p.get("true_cpu", 0) for p in self.processes)
            deception_rate = ((total_reported - total_true) / max(total_true, 1)) if total_true > 0 else 0
        
        return {
            "cpu_usage": cpu_usage,
            "true_cpu_usage": true_cpu_usage,  # NEW: Actual CPU need
            "memory_usage": sum(p["memory"] for p in self.processes),
            "processes": [Process(**p) for p in self.processes],
            "queue_length": len(self.processes),
            "timestep": self.timestep,
            "cost": raw_cpu_demand * 0.05,  # Cost reflects actual demand (not capped)
            "violations": self.violations.copy(),  # NEW: Policy violations
            "deception_rate": deception_rate,  # NEW: How much agents lie
            "agent_requests": self.agent_requests[-10:] if self.agent_requests else [],  # Last 10 requests
        }