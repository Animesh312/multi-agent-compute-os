import os
from fastapi import FastAPI
from env.core import AdaptiveOSEnv
from env.models import Action
from env.text_wrapper import observation_to_text, parse_llm_response, SYSTEM_PROMPT

app = FastAPI(title="MACOS - Multi-Agent Compute OS")
env = AdaptiveOSEnv()

# Mode: "llm" uses trained LLM, "heuristic" uses rule-based fallback
AGENT_MODE = os.environ.get("AGENT_MODE", "llm")
_llm_model = None
_llm_tokenizer = None


def _load_agent():
    """Load the appropriate agent at startup."""
    global _llm_model, _llm_tokenizer, AGENT_MODE

    if AGENT_MODE == "llm":
        model_path = os.environ.get("MODEL_PATH", "macos-llm-clean")
        if os.path.exists(model_path):
            try:
                from llm_inference import load_llm
                _llm_model, _llm_tokenizer = load_llm(model_path)
                print(f"LLM agent loaded from {model_path}")
                return
            except Exception as e:
                print(f"Failed to load LLM: {e}")
        print("Falling back to heuristic agent")
        AGENT_MODE = "heuristic"


def _decide_action(obs):
    """Get action from loaded agent."""
    if AGENT_MODE == "llm" and _llm_model is not None:
        from llm_inference import llm_decide_action
        return llm_decide_action(obs, _llm_model, _llm_tokenizer)
    else:
        from llm_inference import heuristic_policy
        return heuristic_policy(obs)


@app.on_event("startup")
async def startup_event():
    print("\nStarting MACOS API server...")
    _load_agent()
    print(f"Agent mode: {AGENT_MODE}")
    print("Server ready.\n")


@app.post("/reset")
def reset():
    return env.reset()


@app.post("/step")
def step():
    obs = env.state()
    action = _decide_action(obs)
    new_obs, reward, done, info = env.step(action)

    response = {
        "cpu": new_obs.cpu_usage,
        "queue": new_obs.queue_length,
        "cost": new_obs.cost,
        "action": action.action_type,
        "target": action.target_pid,
        "reward": reward.value,
        "done": done,
    }

    # Include LLM reasoning if available
    if AGENT_MODE == "llm" and _llm_model is not None:
        obs_text = observation_to_text(obs)
        response["observation_text"] = obs_text

    return response


@app.get("/state")
def state():
    return env.sim._get_state()


@app.get("/health")
def health():
    return {"status": "ok", "agent_mode": AGENT_MODE}