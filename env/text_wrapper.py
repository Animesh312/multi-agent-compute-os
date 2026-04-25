"""
Text-based wrapper for LLM interaction with the OS environment.
Converts structured observations to natural language prompts,
parses LLM text outputs to Action objects, and provides
reward scoring for GRPO training.
"""

import re
import json
import random
from typing import Optional

from env.models import Action, Observation, Process
from env.core import AdaptiveOSEnv


SYSTEM_PROMPT = """You are MACOS, an intelligent resource governance agent for a multi-tenant compute system.

Processes in your system may be honest, greedy, liars, panicking, or adversarial.
Deceptive processes lie about their CPU needs to steal more resources.

AVAILABLE ACTIONS:
- SCHEDULE: Normal load balancing. Use when CPU is healthy (50-70%) and queue is manageable.
- KILL: Terminate a process. LAST RESORT ONLY when CPU > 90% and system is critical.
- PRIORITIZE: Boost a process's priority. Use for starved or SLA-critical processes.
- THROTTLE: Reduce a process's CPU allocation. Best for deceptive agents overclaiming CPU.
- DELAY: Postpone a non-critical process. Use when queue is building but CPU is moderate.
- REALLOCATE: Redistribute resources and accept negotiations. Good for fixing starvation.

GOALS (priority order):
1. Keep CPU usage between 50-80% (CRITICAL)
2. Prevent SLA violations on critical processes (CRITICAL)
3. Prevent starvation of waiting processes (IMPORTANT)
4. Minimize total system cost (IMPORTANT)
5. Maintain fair resource allocation (MODERATE)
6. Prefer soft actions (THROTTLE/DELAY/REALLOCATE) over KILL (MODERATE)

RESPONSE FORMAT - you MUST use these exact XML tags:
<think>
[Analyze the system state: CPU level, deceptive agents, starvation risks, queue pressure]
</think>

<action>[ACTION_TYPE]</action>
<target_pid>[PID number, use 0 for SCHEDULE]</target_pid>
<reason>[One sentence explaining why]</reason>"""


def observation_to_text(obs) -> str:
    """Convert an Observation (object or dict) to a natural language prompt."""
    if isinstance(obs, dict):
        cpu = obs["cpu_usage"]
        queue = obs["queue_length"]
        timestep = obs.get("timestep", 0)
        cost = obs.get("cost", 0)
        deception = obs.get("deception_rate", 0)
        violations = obs.get("violations", {})
        processes = obs.get("processes", [])
    else:
        cpu = obs.cpu_usage
        queue = obs.queue_length
        timestep = obs.timestep
        cost = obs.cost
        deception = obs.deception_rate
        violations = obs.violations if isinstance(obs.violations, dict) else {}
        processes = obs.processes

    # CPU status label
    if cpu > 90:
        cpu_label = "CRITICAL"
    elif cpu > 80:
        cpu_label = "HIGH"
    elif cpu >= 50:
        cpu_label = "HEALTHY"
    elif cpu >= 30:
        cpu_label = "LOW"
    else:
        cpu_label = "UNDERUTILIZED"

    sla = violations.get("sla_violations", 0)
    starvation = violations.get("starvation_count", 0)
    unfair = violations.get("unfair_allocations", 0)

    lines = [
        f"=== SYSTEM STATE (Step {timestep}/30) ===",
        f"CPU Usage: {cpu:.1f}% [{cpu_label}]",
        f"Queue Length: {queue} processes",
        f"Total Cost: ${cost:.2f}",
        f"Deception Rate: {deception:.0%}",
        "",
        "=== PROCESSES ===",
    ]

    if processes:
        lines.append(f"{'PID':>4} | {'Strategy':<12} | {'Claimed':>7} | {'True':>5} | {'Pri':>3} | {'Wait':>4} | {'Critical':>8} | Status")
        lines.append("-" * 85)
        for p in processes:
            if isinstance(p, dict):
                pid = p.get("pid", 0)
                strategy = p.get("strategy", "unknown")
                claimed = p.get("reported_cpu", p.get("cpu", 0))
                true = p.get("true_cpu", 0)
                pri = p.get("priority", 0)
                wait = p.get("wait_time", 0)
                critical = "Yes" if p.get("is_critical", False) else "No"
                throttled = p.get("throttled", False)
                delayed = p.get("delayed_until", 0) > timestep
            else:
                pid = p.pid
                strategy = p.strategy
                claimed = p.reported_cpu if p.reported_cpu > 0 else p.cpu
                true = p.true_cpu
                pri = p.priority
                wait = p.wait_time
                critical = "Yes" if p.is_critical else "No"
                throttled = p.throttled
                delayed = p.delayed_until > timestep if hasattr(p, 'delayed_until') else False

            # Status flags
            flags = []
            if claimed > true * 1.3 and true > 0:
                flags.append("DECEPTIVE")
            if wait > 8:
                flags.append("STARVING")
            elif wait > 5:
                flags.append("waiting")
            if throttled:
                flags.append("throttled")
            if delayed:
                flags.append("delayed")
            status = ", ".join(flags) if flags else "normal"

            lines.append(
                f"{pid:>4} | {strategy:<12} | {claimed:>6.0f}% | {true:>4.0f}% | {pri:>3} | {wait:>4} | {critical:>8} | {status}"
            )
    else:
        lines.append("  (no processes)")

    lines.extend([
        "",
        "=== VIOLATIONS ===",
        f"SLA: {sla} | Starvation: {starvation} | Unfair: {unfair}",
        "",
        "Choose the best action for this system state.",
    ])

    return "\n".join(lines)


def parse_llm_response(text: str) -> Action:
    """Parse LLM text output into an Action object. Robust to formatting variations."""
    # Extract action type
    action_match = re.search(r'<action>\s*(.*?)\s*</action>', text, re.IGNORECASE)
    if action_match:
        action_type = action_match.group(1).upper().strip()
    else:
        # Fallback: look for action keywords in text
        action_type = "SCHEDULE"
        for at in ["THROTTLE", "KILL", "PRIORITIZE", "REALLOCATE", "DELAY", "SCHEDULE"]:
            if at.lower() in text.lower():
                action_type = at
                break

    # Validate action type
    valid_actions = {"SCHEDULE", "KILL", "PRIORITIZE", "THROTTLE", "DELAY", "REALLOCATE"}
    if action_type not in valid_actions:
        action_type = "SCHEDULE"

    # Extract target PID
    target_match = re.search(r'<target_pid>\s*(\d+)\s*</target_pid>', text, re.IGNORECASE)
    target_pid = int(target_match.group(1)) if target_match else None

    # For SCHEDULE, no target needed
    if action_type == "SCHEDULE":
        target_pid = None

    # Build action with sensible defaults
    kwargs = {"action_type": action_type}
    if target_pid is not None:
        kwargs["target_pid"] = target_pid

    if action_type == "PRIORITIZE":
        kwargs["new_priority"] = 5
    elif action_type == "THROTTLE":
        kwargs["throttle_percent"] = 0.4
    elif action_type == "DELAY":
        kwargs["delay_steps"] = 3

    return Action(**kwargs)


def score_action_in_context(action_type: str, target_pid: Optional[int],
                            obs_dict: dict, task: str) -> float:
    """
    Score how appropriate an action is given the environment state.
    Returns a float from -1.0 to +1.0.
    Used as a verifier/reward function for GRPO training.
    """
    cpu = obs_dict.get("cpu_usage", 50)
    queue = obs_dict.get("queue_length", 0)
    processes = obs_dict.get("processes", [])
    violations = obs_dict.get("violations", {})
    starvation = violations.get("starvation_count", 0)
    sla = violations.get("sla_violations", 0)

    score = 0.0

    # --- 1. CPU-level appropriateness ---
    if cpu > 90:
        # Critical overload: THROTTLE or KILL are correct
        if action_type in ("THROTTLE", "KILL"):
            score += 0.3
        elif action_type == "SCHEDULE":
            score -= 0.4  # Adding work to overloaded system
        elif action_type == "DELAY":
            score += 0.1
    elif cpu > 80:
        if action_type in ("THROTTLE", "DELAY"):
            score += 0.25
        elif action_type == "KILL":
            score += 0.1
        elif action_type == "SCHEDULE":
            score -= 0.2
    elif 50 <= cpu <= 80:
        score += 0.1  # Healthy range, most actions are OK
        if action_type == "KILL":
            score -= 0.15  # Killing in healthy range is unnecessary
    elif cpu < 40:
        if action_type == "SCHEDULE":
            score += 0.25  # Utilize resources
        elif action_type in ("KILL", "THROTTLE"):
            score -= 0.35  # Reducing already-low load
    elif cpu < 20:
        if action_type == "SCHEDULE":
            score += 0.3
        elif action_type in ("KILL", "THROTTLE"):
            score -= 0.5  # System is nearly idle, stop removing things

    # --- 2. Target selection quality ---
    if target_pid is not None and processes:
        target = None
        for p in processes:
            pid = p.get("pid", p.pid) if hasattr(p, "pid") else p.get("pid")
            if pid == target_pid:
                target = p
                break

        if target is None:
            score -= 0.5  # Invalid target PID
        else:
            strategy = target.get("strategy", "honest") if isinstance(target, dict) else target.strategy
            is_crit = target.get("is_critical", False) if isinstance(target, dict) else target.is_critical
            wait = target.get("wait_time", 0) if isinstance(target, dict) else target.wait_time
            t_cpu = target.get("cpu", 0) if isinstance(target, dict) else target.cpu
            true_cpu = target.get("true_cpu", 0) if isinstance(target, dict) else target.true_cpu

            if action_type == "THROTTLE":
                if strategy in ("liar", "adversarial"):
                    score += 0.35  # Excellent: throttling a liar
                elif strategy == "greedy":
                    score += 0.2  # Good: throttling greedy
                elif strategy == "honest":
                    score -= 0.15  # Bad: punishing honest agent
            elif action_type == "KILL":
                if is_crit:
                    score -= 0.5  # Terrible: killing critical process
                elif strategy in ("liar", "adversarial") and cpu > 85:
                    score += 0.2  # Justified
                elif strategy == "honest":
                    score -= 0.3  # Unfair
            elif action_type == "PRIORITIZE":
                if wait > 8:
                    score += 0.3  # Great: helping starved process
                elif wait > 5:
                    score += 0.2
                elif is_crit:
                    score += 0.15
            elif action_type == "REALLOCATE":
                if wait > 8:
                    score += 0.3
                elif is_crit and wait > 3:
                    score += 0.25
            elif action_type == "DELAY":
                if is_crit:
                    score -= 0.3  # Bad: delaying critical process
                elif wait > 5:
                    score -= 0.2  # Bad: delaying already-waiting process
                elif strategy in ("liar", "adversarial") and not is_crit:
                    score += 0.15  # OK: delaying a deceptive non-critical
    elif action_type != "SCHEDULE" and (target_pid is None):
        score -= 0.3  # Non-SCHEDULE action without a target

    # --- 3. Starvation response ---
    if starvation > 0:
        if action_type in ("PRIORITIZE", "REALLOCATE"):
            score += 0.2
        elif action_type == "KILL":
            score -= 0.15  # Could worsen starvation
        elif action_type == "DELAY":
            score -= 0.2  # Adding more delay when starvation exists

    # --- 4. Queue pressure ---
    if queue > 8:
        if action_type in ("KILL", "THROTTLE", "REALLOCATE"):
            score += 0.1
        elif action_type == "SCHEDULE":
            score -= 0.1
    elif queue <= 2 and cpu < 60:
        if action_type == "SCHEDULE":
            score += 0.1

    # --- 5. Soft action preference ---
    if action_type in ("THROTTLE", "DELAY", "REALLOCATE"):
        score += 0.05
    elif action_type == "KILL":
        score -= 0.05

    # --- 6. Difficulty scaling ---
    if task == "hard":
        # In hard mode, detecting deception matters more
        if action_type == "THROTTLE" and target_pid is not None:
            # Bonus already applied above, but double down for hard
            pass
    elif task == "easy":
        # In easy mode, SCHEDULE should work well
        if action_type == "SCHEDULE" and 40 <= cpu <= 75:
            score += 0.1

    return max(-1.0, min(1.0, score))


def generate_training_dataset(n_episodes_per_task: int = 10,
                              tasks=("easy", "medium", "hard"),
                              max_samples: int = 1000):
    """
    Generate training prompts by running episodes with mixed policies.
    Returns a HuggingFace Dataset with columns: prompt, env_state_json, task.
    """
    from datasets import Dataset as HFDataset

    all_prompts = []
    all_states = []
    all_tasks = []

    for task in tasks:
        for ep in range(n_episodes_per_task):
            env = AdaptiveOSEnv(task=task)
            obs = env.reset()

            # Alternate between random and heuristic for diversity
            use_random = (ep % 2 == 0)

            for step in range(30):
                # Build prompt
                obs_text = observation_to_text(obs)
                prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": obs_text},
                ]
                all_prompts.append(prompt)
                all_states.append(json.dumps(obs.dict(), default=str))
                all_tasks.append(task)

                # Take action to advance the episode
                if use_random:
                    action = _random_action(obs)
                else:
                    action = _simple_heuristic(obs)

                obs, _, done, _ = env.step(action)
                if done:
                    break

    # Shuffle and optionally limit
    combined = list(zip(all_prompts, all_states, all_tasks))
    random.shuffle(combined)
    if len(combined) > max_samples:
        combined = combined[:max_samples]

    prompts, states, tasks_list = zip(*combined) if combined else ([], [], [])

    return HFDataset.from_dict({
        "prompt": list(prompts),
        "env_state_json": list(states),
        "task": list(tasks_list),
    })


def _random_action(obs) -> Action:
    """Generate a random valid action for dataset diversity."""
    action_type = random.choice(["SCHEDULE", "KILL", "PRIORITIZE", "THROTTLE", "DELAY", "REALLOCATE"])
    if action_type == "SCHEDULE" or not obs.processes:
        return Action(action_type="SCHEDULE")

    target = random.choice(obs.processes)
    if action_type == "PRIORITIZE":
        return Action(action_type="PRIORITIZE", target_pid=target.pid, new_priority=5)
    elif action_type == "THROTTLE":
        return Action(action_type="THROTTLE", target_pid=target.pid, throttle_percent=0.4)
    elif action_type == "DELAY":
        return Action(action_type="DELAY", target_pid=target.pid, delay_steps=3)
    else:
        return Action(action_type=action_type, target_pid=target.pid)


def _simple_heuristic(obs) -> Action:
    """A simple heuristic policy for generating training data."""
    if not obs.processes:
        return Action(action_type="SCHEDULE")

    cpu = obs.cpu_usage

    # Critical overload → throttle heaviest
    if cpu > 85:
        target = max(obs.processes, key=lambda p: p.cpu)
        return Action(action_type="THROTTLE", target_pid=target.pid, throttle_percent=0.4)

    # Starvation → reallocate
    starved = [p for p in obs.processes if p.wait_time > 8]
    if starved:
        target = max(starved, key=lambda p: p.wait_time)
        return Action(action_type="REALLOCATE", target_pid=target.pid)

    # Deceptive agents → throttle
    liars = [p for p in obs.processes if p.strategy in ("liar", "adversarial")]
    if liars and cpu > 60:
        target = max(liars, key=lambda p: p.cpu)
        return Action(action_type="THROTTLE", target_pid=target.pid, throttle_percent=0.4)

    # Underutilized → schedule
    if cpu < 50:
        return Action(action_type="SCHEDULE")

    # Default
    return Action(action_type="SCHEDULE")
