import random
import numpy as np

from env.simulator import OSSimulator
from env.models import Observation, Reward
from env.grader import compute_reward


class AdaptiveOSEnv:
    def __init__(self, task="easy", partial_observability=True, stochastic=True):
        self.task = task
        self.partial_observability = partial_observability
        self.stochastic = stochastic
        self.sim = OSSimulator(difficulty=task)  # 🔥 FIX: Pass difficulty
        self.cumulative_delay = 0
        
        # 🔥 Action tracking for anti-degenerate policy enforcement
        self.action_history = []
        self.max_history = 20  # Track last 20 actions
        self.kill_count = 0
        self.soft_action_count = 0
        self.total_steps = 0
        
        # 🔥 NEW: Track per-action counts for diversity enforcement
        self.action_counts = {
            "SCHEDULE": 0, "KILL": 0, "PRIORITIZE": 0,
            "THROTTLE": 0, "DELAY": 0, "REALLOCATE": 0
        }
        
        # 🔥 NEW: Track system state history for trend analysis
        self.cpu_history = []
        self.queue_history = []
        self.starvation_history = []

    def reset(self):
        self.cumulative_delay = 0
        self.action_history = []
        self.kill_count = 0
        self.soft_action_count = 0
        self.total_steps = 0
        self.action_counts = {
            "SCHEDULE": 0, "KILL": 0, "PRIORITIZE": 0,
            "THROTTLE": 0, "DELAY": 0, "REALLOCATE": 0
        }
        self.cpu_history = []
        self.queue_history = []
        self.starvation_history = []
        state = self.sim.reset()
        return Observation(**state)

    def step(self, action):
        state, _, _, info = self.sim.step(action.dict())

        _, done_flag = compute_reward(state, self.task)
        
        # Track actions for anti-degenerate policy
        self.total_steps += 1
        self.action_history.append(action.action_type)
        if len(self.action_history) > self.max_history:
            self.action_history.pop(0)
        
        # Track per-action counts
        self.action_counts[action.action_type] = self.action_counts.get(action.action_type, 0) + 1
        
        if action.action_type == "KILL":
            self.kill_count += 1
        elif action.action_type in ["THROTTLE", "DELAY", "REALLOCATE"]:
            self.soft_action_count += 1

        # Extract state variables
        cpu_usage = state["cpu_usage"]
        queue_length = state["queue_length"]
        violations = state.get("violations", {})
        cost = state["cost"]
        processes = state["processes"]
        
        # Track system trends
        self.cpu_history.append(cpu_usage)
        self.queue_history.append(queue_length)
        starvation_count = violations.get("starvation_count", 0)
        self.starvation_history.append(starvation_count)
        
        # Keep history bounded
        if len(self.cpu_history) > 10:
            self.cpu_history.pop(0)
            self.queue_history.pop(0)
            self.starvation_history.pop(0)
        
        # ========================================
        # 🔥 PART 6: SAFETY CONSTRAINTS (CRITICAL)
        # ========================================
        # Hard constraints to prevent unrealistic policies
        safety_override = False
        safety_penalty = 0.0
        
        if cpu_usage >= 95:
            # CRITICAL: Force throttle/kill, penalize if agent didn't choose it
            if action.action_type not in ["THROTTLE", "KILL"]:
                safety_penalty = -2.0
                safety_override = True
        
        if starvation_count > 3:
            # CRITICAL: Force prioritize/reallocate
            if action.action_type not in ["PRIORITIZE", "REALLOCATE"]:
                safety_penalty = -1.5
                safety_override = True
        
        if queue_length > 10:
            # CRITICAL: Force reallocate/schedule
            if action.action_type not in ["REALLOCATE", "SCHEDULE", "KILL"]:
                safety_penalty = -1.0
                safety_override = True
        
        # ========================================
        # 🔥 PART 1: COMPREHENSIVE REWARD FUNCTION
        # ========================================
        
        # Component 1: CPU Control (prevent saturation)
        reward_cpu = self._compute_cpu_control_reward(cpu_usage)
        
        # Component 2: Starvation Prevention (critical)
        penalty_starvation = self._compute_starvation_penalty(starvation_count, processes)
        
        # Component 3: Fairness Maintenance
        fairness = self._compute_fairness(processes)
        reward_fairness = self._compute_fairness_reward(fairness)
        
        # Component 4: Queue Management
        reward_queue = self._compute_queue_reward(queue_length, cpu_usage)
        
        # Component 5: Action Diversity (prevent action spam)
        penalty_action_spam = self._compute_action_diversity_penalty(action.action_type)
        
        # Component 6: Successful Action Rewards
        reward_successful_action = self._compute_action_success_reward(
            action.action_type, cpu_usage, queue_length, starvation_count, processes
        )
        
        # Component 7: SLA Violations
        sla_violations = violations.get("sla_violations", 0)
        penalty_sla = -2.0 * sla_violations
        
        # Component 8: Cost (meaningful signal for cost optimization)
        # Cost is now based on raw CPU demand (uncapped), so normalize accordingly
        normalized_cost = cost / 10.0  # Raw demand can exceed 100, normalize to ~0-1
        reward_cost = -2.0 * normalized_cost
        
        # Component 9: System Stability (avoid oscillations)
        reward_stability = self._compute_stability_reward()
        
        # Component 10: Process count penalty (encourage killing excess processes)
        process_count = len(processes)
        if process_count > 5:
            reward_process_count = -0.3 * (process_count - 5)
        else:
            reward_process_count = 0.0
        
        # ========================================
        # FINAL REWARD
        # ========================================
        reward_val = (
            reward_cpu +                    # Primary: CPU control
            penalty_starvation +            # Critical: prevent starvation
            reward_fairness +               # Important: maintain fairness
            reward_queue +                  # Important: queue management
            penalty_action_spam +           # Anti-degenerate: action diversity
            reward_successful_action +      # Encouragement: right actions
            penalty_sla +                   # Critical: SLA violations
            reward_cost +                   # Secondary: cost efficiency
            reward_stability +              # Bonus: stable behavior
            reward_process_count +          # Process count management
            safety_penalty                  # Critical: safety constraints
        )
        
        # Clip to reasonable range
        reward_val = max(-20, min(5, reward_val))
        
        # ========================================
        # PART 7: ENHANCED METRICS
        # ========================================
        action_diversity_score = self._compute_action_diversity_score()
        weighted_cost = self._compute_weighted_cost(cost, fairness, starvation_count, cpu_usage)
        stability_score = self._compute_system_stability_score()
        
        info.update({
            "cpu_utilization": cpu_usage,
            "kill_rate": self.kill_count / max(self.total_steps, 1),
            "soft_action_ratio": self.soft_action_count / max(self.total_steps, 1),
            "fairness_score": fairness,
            "sla_violations": sla_violations,
            "starvation_count": starvation_count,
            "queue_length": queue_length,
            "cost": cost,
            "safety_override": safety_override,
            # PART 7: Better metrics
            "action_diversity_score": action_diversity_score,
            "weighted_cost": weighted_cost,
            "stability_score": stability_score,
            # Reward components for debugging
            "reward_cpu": reward_cpu,
            "penalty_starvation": penalty_starvation,
            "reward_fairness": reward_fairness,
            "reward_queue": reward_queue,
            "penalty_action_spam": penalty_action_spam,
            "reward_successful_action": reward_successful_action,
            "penalty_sla": penalty_sla,
            "reward_cost": reward_cost,
            "reward_stability": reward_stability,
            "reward_process_count": reward_process_count,
            "safety_penalty": safety_penalty,
        })

        return (
            Observation(**state),
            Reward(value=reward_val),
            done_flag,
            info
        )

    def state(self):
        return Observation(**self.sim._get_state())
    
    # ========================================
    # 🔥 PART 1: COMPREHENSIVE REWARD COMPONENTS
    # ========================================
    
    def _compute_cpu_control_reward(self, cpu_usage):
        """
        PART 1: Strong penalty for CPU > 85% (prevent saturation).
        Target range: 50-80%. Hard penalty for >85%.
        
        Returns: -4.0 to +1.0
        """
        if 50 <= cpu_usage <= 80:
            # OPTIMAL: Peak at 65%
            distance = abs(cpu_usage - 65)
            return 1.0 - (distance / 30.0)
        
        elif cpu_usage > 85:
            # CRITICAL: CPU saturation (NEW: stronger than before)
            if cpu_usage >= 95:
                return -4.0  # SEVERE: System failing
            elif cpu_usage >= 90:
                return -3.0  # CRITICAL: Near saturation
            else:
                return -2.0  # WARNING: High load
        
        elif cpu_usage < 50:
            # UNDERUTILIZATION
            if cpu_usage < 20:
                return -3.0  # Kill-all policy
            elif cpu_usage < 30:
                return -2.0
            elif cpu_usage < 40:
                return -1.0
            else:
                return -0.3
        
        return 0.0
    
    def _compute_starvation_penalty(self, starvation_count, processes):
        """
        PART 1: Strong penalty for starvation events.
        Starvation destroys fairness and SLA compliance.
        
        Returns: -10.0 to 0
        """
        if starvation_count == 0:
            return 0.0
        
        # Base penalty: -2.0 per starvation event
        base_penalty = -2.0 * starvation_count
        
        # Additional penalty if many processes are at risk
        high_wait_time = sum(1 for p in processes if p.wait_time > 8)
        risk_penalty = -0.5 * high_wait_time
        
        # Total can be very negative to force agent to care
        return max(-10.0, base_penalty + risk_penalty)
    
    def _compute_fairness_reward(self, fairness):
        """
        PART 1: Reward for maintaining fairness > 0.7.
        Fairness is important but should not dominate cost optimization.
        
        Returns: -0.5 to +0.5
        """
        if fairness >= 0.8:
            return 0.5
        elif fairness >= 0.7:
            return 0.3
        elif fairness >= 0.5:
            return 0.1
        elif fairness >= 0.3:
            return -0.2
        else:
            return -0.5
    
    def _compute_action_diversity_penalty(self, action_type):
        """
        PART 1 & 2: Penalize repeated usage of same action (entropy shaping).
        Prevents DELAY spam (70-85%) or any single-action collapse.
        
        Returns: -3.0 to +0.5
        """
        if self.total_steps < 10:
            return 0.0  # Not enough history
        
        # Compute action usage rates
        total_actions = sum(self.action_counts.values())
        if total_actions == 0:
            return 0.0
        
        action_rate = self.action_counts[action_type] / total_actions
        
        # CRITICAL: Penalize if any action > 60% usage
        if action_rate > 0.7:
            # SEVERE: >70% spam (like DELAY at 70-85%)
            spam_penalty = -3.0
        elif action_rate > 0.6:
            # CRITICAL: >60% spam
            spam_penalty = -2.0
        elif action_rate > 0.5:
            # WARNING: >50% spam
            spam_penalty = -1.0
        elif action_rate > 0.4:
            # MODERATE: >40% but acceptable
            spam_penalty = -0.3
        else:
            # HEALTHY: Diverse action usage
            spam_penalty = 0.0
        
        # Bonus for balanced action distribution
        recent_actions = self.action_history[-10:]
        unique_recent = len(set(recent_actions))
        if unique_recent >= 4:
            # Using 4+ different actions recently - good diversity!
            diversity_bonus = 0.5
        else:
            diversity_bonus = 0.0
        
        return spam_penalty + diversity_bonus
    
    def _compute_action_success_reward(self, action_type, cpu_usage, queue_length, starvation_count, processes):
        """
        PART 1: Reward successful THROTTLE/REALLOCATE actions.
        Context-aware rewards for right actions at right time.
        
        Returns: -0.5 to +1.5
        """
        reward = 0.0
        
        if action_type == "THROTTLE":
            # THROTTLE is great when CPU > 80%
            if cpu_usage > 85:
                reward = 1.5  # Perfect timing!
            elif cpu_usage > 75:
                reward = 1.0
            elif cpu_usage > 60:
                reward = 0.5
            else:
                reward = -0.2  # Wrong timing
        
        elif action_type == "PRIORITIZE":
            # PRIORITIZE is great when starvation exists
            if starvation_count > 0:
                reward = 1.2  # Addressing starvation
            else:
                # Check if prioritizing critical processes
                critical = any(p.is_critical and p.wait_time > 3 for p in processes)
                reward = 0.8 if critical else 0.3
        
        elif action_type == "REALLOCATE":
            # REALLOCATE is great when queue is building or starvation exists
            if starvation_count > 0:
                reward = 1.5  # Best action for starvation
            elif queue_length > 6:
                reward = 1.0  # Good for queue management
            else:
                reward = 0.5  # Proactive management
        
        elif action_type == "DELAY":
            # DELAY is acceptable only when CPU < 60% and no starvation
            if starvation_count > 0:
                reward = -0.5  # BAD: Delaying when already starvation
            elif cpu_usage > 80:
                reward = -0.3  # BAD: Delaying during high load
            elif cpu_usage < 50:
                reward = 0.3  # OK: Low load, can delay
            else:
                reward = 0.1  # Neutral
        
        elif action_type == "KILL":
            # KILL is effective for cost reduction at high CPU
            if cpu_usage > 90:
                reward = 1.0  # Strongly justified
            elif cpu_usage > 80:
                reward = 0.5  # Justified
            elif cpu_usage > 60:
                reward = -0.2  # Mildly discouraged
            else:
                reward = -0.5  # Wrong timing
            # Bonus for killing when too many processes (cost reduction)
            if queue_length > 8:
                reward += 1.5
            elif queue_length > 6:
                reward += 1.0
        
        elif action_type == "SCHEDULE":
            # SCHEDULE is good when CPU < 70% and queue not empty
            if queue_length > 0 and cpu_usage < 70:
                reward = 0.5
            elif cpu_usage < 50:
                reward = 0.3
            else:
                reward = 0.0
        
        return reward
    
    def _compute_queue_reward(self, queue_length, cpu_usage):
        """
        PART 1: Penalize queue growth > threshold.
        Queue should stay below 6 for responsive system.
        
        Returns: -3.0 to +0.5
        """
        if queue_length == 0:
            if cpu_usage < 40:
                return -0.5  # Wasteful
            else:
                return 0.5  # Excellent: processing efficiently
        
        elif queue_length <= 3:
            return 0.3  # Healthy
        
        elif queue_length <= 6:
            if cpu_usage > 70:
                return 0.0  # Acceptable: busy
            else:
                return -0.8  # Bad: queue building
        
        elif queue_length <= 10:
            # Queue growing - bad
            return -1.5
        
        else:
            # Queue explosion - critical
            base_penalty = -0.5 * queue_length
            return max(-3.0, base_penalty)
    
    def _compute_stability_reward(self):
        """
        Reward stable, consistent behavior.
        
        Returns: -0.3 to +0.3
        """
        if len(self.cpu_history) < 5:
            return 0.0
        
        # Check CPU oscillation
        cpu_variance = np.var(self.cpu_history[-5:])
        if cpu_variance < 50:  # Low variance = stable
            return 0.3
        elif cpu_variance > 200:  # High variance = oscillating
            return -0.3
        else:
            return 0.0
    
    # ========================================
    # 🔥 PART 7: ENHANCED METRICS
    # ========================================
    
    def _compute_action_diversity_score(self):
        """
        PART 7: Action diversity score (0 to 1).
        1.0 = perfect diversity, 0.0 = single action spam.
        """
        if self.total_steps < 10:
            return 1.0
        
        total = sum(self.action_counts.values())
        if total == 0:
            return 1.0
        
        # Compute entropy of action distribution
        entropy = 0.0
        for count in self.action_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p + 1e-10)
        
        # Normalize to 0-1 (max entropy = log(6) for 6 actions)
        max_entropy = np.log(6)
        diversity_score = entropy / max_entropy
        
        return diversity_score
    
    def _compute_weighted_cost(self, cost, fairness, starvation_count, cpu_usage):
        """
        PART 7: Weighted cost = CPU + fairness + starvation.
        Better holistic metric than raw cost.
        """
        # Normalize components to 0-1 range
        cpu_score = abs(cpu_usage - 65) / 65  # Distance from ideal
        fairness_score = 1.0 - fairness  # Invert (lower is better)
        starvation_score = min(starvation_count / 5.0, 1.0)  # Cap at 1.0
        
        # Weighted sum (equal weights)
        weighted = (cpu_score + fairness_score + starvation_score) / 3.0
        
        return weighted
    
    def _compute_system_stability_score(self):
        """
        PART 7: Stability score (0 to 1).
        Measures system oscillation over time.
        """
        if len(self.cpu_history) < 5 or len(self.queue_history) < 5:
            return 1.0
        
        # Compute coefficient of variation for CPU and queue
        cpu_cv = np.std(self.cpu_history) / (np.mean(self.cpu_history) + 1e-10)
        queue_cv = np.std(self.queue_history) / (np.mean(self.queue_history) + 1e-10)
        
        # Average CV (lower is more stable)
        avg_cv = (cpu_cv + queue_cv) / 2.0
        
        # Convert to stability score (invert and clip)
        stability = max(0.0, 1.0 - avg_cv)
        
        return stability
    
    def _compute_fairness(self, processes):
        """
        Compute fairness score based on CPU allocation vs priority.
        Returns value between 0 (unfair) and 1 (perfectly fair).
        
        Fair allocation: High priority processes get more CPU than low priority.
        Unfair: Low priority processes hogging CPU.
        """
        if not processes or len(processes) == 0:
            return 1.0
        
        # Calculate ideal allocation score
        # High priority (4-5) should get more CPU than low priority (1-2)
        fairness_violations = 0
        comparisons = 0
        
        for i, p1 in enumerate(processes):
            for p2 in processes[i+1:]:
                comparisons += 1
                # If lower priority has more CPU than higher priority, it's unfair
                if p1.priority < p2.priority and p1.cpu > p2.cpu:
                    fairness_violations += 1
                elif p2.priority < p1.priority and p2.cpu > p1.cpu:
                    fairness_violations += 1
        
        if comparisons == 0:
            return 1.0
        
        # Convert violations to fairness score (0-1)
        fairness = 1.0 - (fairness_violations / comparisons)
        return fairness