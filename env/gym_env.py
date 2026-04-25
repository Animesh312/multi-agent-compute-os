import gymnasium as gym
import numpy as np
from env.core import AdaptiveOSEnv
from env.models import Action

class AdaptiveOSGymEnv(gym.Env):
    def __init__(self, task="easy", use_enhanced_obs=False):
        super().__init__()
        self.env = AdaptiveOSEnv(task=task)
        self.prev_queue = 0
        self.prev_cpu = 0
        self.reward_history = []
        
        # 🔥 PART 3: Track history for trends (only if enhanced obs enabled)
        self.use_enhanced_obs = use_enhanced_obs
        self.cpu_history = []
        self.queue_history = []

        # 🔥 UPGRADED: Action space with soft actions
        # 0=SCHEDULE, 1=KILL, 2=PRIORITIZE, 3=THROTTLE, 4=DELAY, 5=REALLOCATE
        self.action_space = gym.spaces.Discrete(6)

        # 🔥 PART 3: IMPROVED State space with trend analysis
        # Base features: cpu_norm, cpu_trend, queue_length, queue_growth, load_level, reward_trend
        # Aggregate stats: avg_wait_time, max_wait_time, starvation_risk, deception_confidence
        # Process features: strategy_onehot (5), cpu_demand_norm, waiting_time_norm, deception_signal
        max_procs = 10
        proc_features = 5 + 1 + 1 + 1  # strategy(5) + cpu + waiting + deception
        
        if use_enhanced_obs:
            # New enhanced observation space (90 dimensions)
            base_features = 6  # cpu_norm, cpu_trend, queue, queue_growth, load, reward_trend
            aggregate_features = 4  # avg_wait, max_wait, starvation_risk, deception_conf
            state_dim = base_features + aggregate_features + max_procs * proc_features
        else:
            # Legacy observation space (84 dimensions) for backward compatibility
            base_features = 4  # cpu_norm, queue_delta, load, reward_trend
            aggregate_features = 0
            state_dim = base_features + max_procs * proc_features
            
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        self.prev_queue = obs.queue_length
        self.prev_cpu = obs.cpu_usage
        self.reward_history = []
        self.cpu_history = []
        self.queue_history = []
        return self._get_state(obs), {}

    def step(self, action):
        # Map action to Action model with soft actions
        state = self.env.state()
        
        if action == 0:
            act = Action(action_type="SCHEDULE")
        elif action == 1:
            # KILL - target highest CPU (but RL should learn to avoid this)
            heaviest = max(state.processes, key=lambda p: p.cpu, default=None)
            act = Action(action_type="KILL", target_pid=heaviest.pid if heaviest else 0)
        elif action == 2:
            # PRIORITIZE - boost lowest priority
            lowest = min(state.processes, key=lambda p: p.priority, default=None)
            act = Action(action_type="PRIORITIZE", target_pid=lowest.pid if lowest else 0, new_priority=5)
        elif action == 3:
            # 🔥 THROTTLE - reduce CPU for highest-consuming process
            # Prefer deceptive agents, but fall back to highest CPU consumer
            deceptive = [p for p in state.processes if p.strategy in ["liar", "greedy", "adversarial"]]
            if deceptive:
                target = max(deceptive, key=lambda p: p.cpu)
                act = Action(action_type="THROTTLE", target_pid=target.pid, throttle_percent=0.3)
            elif state.processes:
                target = max(state.processes, key=lambda p: p.cpu)
                act = Action(action_type="THROTTLE", target_pid=target.pid, throttle_percent=0.4)
            else:
                act = Action(action_type="SCHEDULE")
        elif action == 4:
            # 🔥 DELAY - delay highest-CPU non-critical process
            delayable = [p for p in state.processes if not p.is_critical]
            if delayable:
                target = max(delayable, key=lambda p: p.cpu)
                act = Action(action_type="DELAY", target_pid=target.pid, delay_steps=3)
            else:
                act = Action(action_type="SCHEDULE")
        elif action == 5:
            # 🔥 REALLOCATE - boost underserved process (starved or low priority)
            starved = [p for p in state.processes if p.wait_time > 5]
            if starved:
                target = max(starved, key=lambda p: p.wait_time)
                act = Action(action_type="REALLOCATE", target_pid=target.pid)
            elif state.processes:
                lowest = min(state.processes, key=lambda p: p.priority)
                act = Action(action_type="REALLOCATE", target_pid=lowest.pid)
            else:
                act = Action(action_type="SCHEDULE")
        else:
            act = Action(action_type="SCHEDULE")

        obs, reward, done, info = self.env.step(act)
        state = self._get_state(obs)
        self.reward_history.append(reward.value)
        # Gymnasium expects 5 values: (obs, reward, terminated, truncated, info)
        return state, reward.value, done, False, info

    def _get_state(self, obs):
        """
        PART 3: Improved observation space with trend analysis and aggregate stats.
        Supports legacy mode (84 dims) and enhanced mode (90 dims).
        
        Why each feature helps learning:
        - CPU trend: Detects if system is getting worse (rising) or better (falling)
        - Queue growth rate: Early warning signal for queue explosion
        - Per-agent wait time stats: Identifies starvation risk before it happens
        - Starvation risk indicator: Binary signal for urgent action needed
        - Deception confidence: Helps identify which processes to throttle/kill
        """
        # Base features
        cpu_norm = obs.cpu_usage / 100.0
        
        queue_length_norm = obs.queue_length / 15.0  # Normalize
        queue_delta = (obs.queue_length - self.prev_queue) / 15.0
        self.prev_queue = obs.queue_length
        
        load_level = 0 if obs.cpu_usage < 40 else 1 if obs.cpu_usage < 80 else 2
        
        reward_trend = np.mean(self.reward_history[-5:]) if self.reward_history else 0
        
        # Enhanced features (only if use_enhanced_obs=True)
        if self.use_enhanced_obs:
            # PART 3: CPU trend (delta CPU)
            self.cpu_history.append(obs.cpu_usage)
            if len(self.cpu_history) > 5:
                self.cpu_history.pop(0)
            cpu_trend = (obs.cpu_usage - self.prev_cpu) / 100.0 if self.prev_cpu > 0 else 0.0
            self.prev_cpu = obs.cpu_usage
            
            # PART 3: Queue growth rate
            self.queue_history.append(obs.queue_length)
            if len(self.queue_history) > 5:
                self.queue_history.pop(0)
            
            # PART 3: Aggregate statistics
            if obs.processes:
                # Per-agent wait time stats
                wait_times = [p.wait_time for p in obs.processes]
                avg_wait_time = np.mean(wait_times) / 15.0  # Normalize
                max_wait_time = max(wait_times) / 15.0
                
                # Starvation risk indicator (how many processes at risk)
                starvation_risk = sum(1 for p in obs.processes if p.wait_time > 8) / len(obs.processes)
                
                # Deception confidence score (how much deception detected)
                deceptions = [abs(p.reported_cpu - p.true_cpu) for p in obs.processes]
                deception_confidence = np.mean(deceptions) / 50.0  # Normalize
            else:
                avg_wait_time = 0.0
                max_wait_time = 0.0
                starvation_risk = 0.0
                deception_confidence = 0.0

        proc_features = []
        all_strategies = ["honest", "greedy", "panic", "liar", "adversarial"]
        
        for p in obs.processes[:10]:  # max 10
            strategy_onehot = [1 if p.strategy == s else 0 for s in all_strategies]
            cpu_demand_norm = p.true_cpu / 50.0
            waiting_time_norm = p.wait_time / 15.0
            deception_signal = (p.reported_cpu - p.true_cpu) / max(p.true_cpu, 1.0)
            
            proc_features.extend(strategy_onehot + [cpu_demand_norm, waiting_time_norm, deception_signal])

        # Pad to max_procs (8 features per process)
        while len(proc_features) < 10 * 8:
            proc_features.extend([0] * 8)

        # Combine all features
        if self.use_enhanced_obs:
            # Enhanced: 90 dimensions
            state = [
                cpu_norm,              # Current CPU
                cpu_trend,             # PART 3: CPU delta (rising/falling)
                queue_length_norm,     # Current queue
                queue_delta,           # Queue delta (was queue_growth)
                load_level,            # Load category
                reward_trend,          # Recent reward trend
                avg_wait_time,         # PART 3: Average wait time
                max_wait_time,         # PART 3: Max wait time (starvation signal)
                starvation_risk,       # PART 3: Fraction at risk
                deception_confidence   # PART 3: Deception level
            ] + proc_features
        else:
            # Legacy: 84 dimensions (backward compatible)
            state = [
                cpu_norm,              # Current CPU
                queue_delta,           # Queue delta
                load_level,            # Load category
                reward_trend           # Recent reward trend
            ] + proc_features
        
        return np.array(state, dtype=np.float32)

    def render(self, mode='human'):
        pass


class RandomDifficultyGymEnv(AdaptiveOSGymEnv):
    """Randomly selects difficulty each episode for mixed training.
    Exposes the agent to EASY/MEDIUM/HARD scenarios so it learns
    difficulty-appropriate strategies (e.g., KILL on EASY, THROTTLE on HARD).
    """

    def __init__(self, use_enhanced_obs=False):
        super().__init__(task="medium", use_enhanced_obs=use_enhanced_obs)
        self.difficulties = ["easy", "medium", "hard"]

    def reset(self, seed=None, options=None):
        difficulty = np.random.choice(self.difficulties)
        self.env.task = difficulty
        self.env.sim.difficulty = difficulty
        return super().reset(seed=seed, options=options)