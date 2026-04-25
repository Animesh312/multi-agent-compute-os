def compute_reward(state, task):
    cpu = state["cpu_usage"]
    queue = state["queue_length"]

    # Normalize helpers
    cpu_target = 70
    cpu_target_max = 100
    cpu_score = max(0.0, 1.0 - abs(cpu - cpu_target) / cpu_target_max)
    queue_score = max(0.0, 1.0 - min(queue / 10.0, 1.0))

    if task == "easy":
        # Focus: just keep CPU stable
        score = cpu_score

    elif task == "medium":
        # Balance CPU + queue
        score = 0.5 * cpu_score + 0.4 * queue_score

    else:  # hard
        # Advanced: stability + throughput + penalties
        penalty = 0.0

        if cpu > 95:
            penalty += 0.2  # overload
        if queue > 8:
            penalty += 0.2  # backlog

        score = 0.5 * cpu_score + 0.3 * queue_score - penalty

    # Clamp score
    score = max(0.0, min(1.0, score))

    # Episode ends
    done = state["timestep"] >= 30

    return score, done