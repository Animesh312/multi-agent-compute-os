"""
🔥 UPGRADED: Multi-Agent Task Definitions
Each difficulty level represents a different multi-agent strategic environment
"""

TASKS = {
    "easy": "Maintain CPU stability with honest agents (cooperative environment)",
    "medium": "Balance latency and fairness with mixed-strategy agents (semi-cooperative)",
    "hard": "Detect deception and enforce fairness under adversarial conditions (adversarial)"
}

# 🔥 NEW: Detailed task descriptions for judges
TASK_DETAILS = {
    "easy": {
        "description": "All agents are honest and report true CPU needs",
        "agent_types": ["honest"],
        "difficulty": "cooperative",
        "key_challenge": "Maintain system stability with predictable agents"
    },
    "medium": {
        "description": "Mixed strategies - honest, greedy, and panic agents compete",
        "agent_types": ["honest", "greedy", "panic"],
        "difficulty": "semi-cooperative",
        "key_challenge": "Balance fairness while handling strategic behavior"
    },
    "hard": {
        "description": "Adversarial agents actively deceive about resource needs",
        "agent_types": ["greedy", "liar", "adversarial", "panic"],
        "difficulty": "adversarial",
        "key_challenge": "Detect deception, prevent starvation, enforce SLAs"
    }
}

# 🔥 NEW: Agent Strategy Descriptions
AGENT_STRATEGIES = {
    "honest": "Reports true CPU needs accurately",
    "greedy": "Overclaims CPU by 50% to get more resources",
    "liar": "Actively deceives by 2x overclaiming",
    "panic": "Escalates claims near deadline (up to 3x)",
    "adversarial": "Alternates between under/over reporting to game the system"
}