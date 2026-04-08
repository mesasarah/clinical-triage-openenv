"""
Baseline inference script for the Clinical Triage Navigator OpenEnv environment.
Uses the OpenAI client to run a model against all three tasks and reports
reproducible scores following the exact [START]/[STEP]/[END] log format.

Usage:
    export API_BASE_URL="https://api.openai.com/v1"
    export MODEL_NAME="gpt-4.1-mini"
    export HF_TOKEN="hf_..."
    python inference.py
"""

import os
import time
from typing import List, Optional

import httpx
from openai import OpenAI

# ──────────────────────────────────────────────
# Environment variables
# ──────────────────────────────────────────────

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME:   str = os.getenv("MODEL_NAME",   "gpt-4.1-mini")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required.")

# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

ENV_BASE_URL      = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK         = "clinical-triage-env"
TEMPERATURE       = 0.0
MAX_TOKENS        = 1024
MAX_STEPS         = 10
SUCCESS_THRESHOLD = 0.6

TASKS = [
    {"id": "task_easy",   "name": "symptom-triage-classification"},
    {"id": "task_medium", "name": "diagnostic-test-ordering"},
    {"id": "task_hard",   "name": "discharge-plan-generation"},
]

SYSTEM_PROMPT = """You are an expert emergency medicine physician.
You will be given a clinical scenario and must respond precisely as instructed.
Follow all formatting instructions exactly. Be accurate, safe, and thorough."""

# ──────────────────────────────────────────────
# Structured logging (exact required format)
# ──────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    action_safe = action.replace("\n", " ").replace("\r", " ")[:200]
    done_str    = "true" if done else "false"
    error_str   = error if error else "null"
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

# ──────────────────────────────────────────────
# Env HTTP helpers
# ──────────────────────────────────────────────

def env_reset(task_id: str) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(content: str) -> dict:
    r = httpx.post(f"{ENV_BASE_URL}/step", json={"content": content}, timeout=60)
    r.raise_for_status()
    return r.json()


def env_state() -> dict:
    r = httpx.get(f"{ENV_BASE_URL}/state", timeout=10)
    r.raise_for_status()
    return r.json()


def wait_for_env(retries: int = 20, delay: float = 3.0) -> None:
    """Poll /health until the server is ready."""
    for attempt in range(retries):
        try:
            r = httpx.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code == 200:
                print(f"[DEBUG] Environment ready after {attempt + 1} attempt(s).", flush=True)
                return
        except Exception:
            pass
        print(f"[DEBUG] Waiting for environment... ({attempt + 1}/{retries})", flush=True)
        time.sleep(delay)
    raise RuntimeError(f"Environment did not become ready at {ENV_BASE_URL}")

# ──────────────────────────────────────────────
# LLM call
# ──────────────────────────────────────────────

def call_model(client: OpenAI, prompt: str, history: List[str]) -> str:
    """Call the LLM and return the response text."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in history[-4:]:  # keep last 4 turns for context
        messages.append({"role": "assistant", "content": h})
    messages.append({"role": "user", "content": prompt})

    try:
        completion = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = messages,
            temperature = TEMPERATURE,
            max_tokens  = MAX_TOKENS,
            stream      = False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else "NON_URGENT"
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return "NON_URGENT"

# ──────────────────────────────────────────────
# Single task runner
# ──────────────────────────────────────────────

def run_task(client: OpenAI, task: dict) -> dict:
    task_id   = task["id"]
    task_name = task["name"]

    rewards:     List[float] = []
    history:     List[str]   = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result      = env_reset(task_id)
        prompt      = result["observation"]["prompt"]
        last_reward = 0.0

        for step_num in range(1, MAX_STEPS + 1):
            if result.get("done", False):
                break

            action = call_model(client, prompt, history)
            result = env_step(action)

            reward   = float(result.get("reward", 0.0))
            done     = bool(result.get("done", False))
            feedback = result.get("info", {}).get("feedback", "")
            error    = None

            rewards.append(reward)
            steps_taken  = step_num
            last_reward  = reward

            log_step(step=step_num, action=action, reward=reward, done=done, error=error)

            history.append(f"Step {step_num} action: {action[:100]}")
            history.append(f"Step {step_num} feedback: {feedback[:100]}")

            # Update prompt for next step (if not done)
            if not done:
                new_obs = result.get("observation", {})
                prompt  = new_obs.get("prompt", prompt)

            if done:
                break

        # Final score = last non-zero reward (single-step task pattern)
        score   = rewards[-1] if rewards else 0.0
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)
        error_msg = str(exc)
        log_end(success=False, steps=steps_taken, score=0.0, rewards=rewards)
        return {"task_id": task_id, "score": 0.0, "success": False, "error": error_msg}

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success}

# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    print("[DEBUG] Waiting for environment to be ready...", flush=True)
    wait_for_env()

    all_results = []
    for task in TASKS:
        print(f"\n[DEBUG] ── Running task: {task['id']} ──", flush=True)
        result = run_task(client, task)
        all_results.append(result)
        time.sleep(1)  # brief pause between tasks

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("BASELINE RESULTS SUMMARY", flush=True)
    print("=" * 60, flush=True)
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  {r['task_id']:<20} score={r['score']:.3f}  [{status}]", flush=True)
    avg = sum(r["score"] for r in all_results) / len(all_results)
    print(f"\n  Overall average score: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
