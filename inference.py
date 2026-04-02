"""
Inference Script — Customer Service RL Environment
===================================================
MANDATORY:
- API_BASE_URL   The API endpoint for the LLM
- MODEL_NAME     The model identifier to use for inference
- HF_TOKEN       Your Hugging Face / API key

Usage:
    API_BASE_URL=https://router.huggingface.co/v1 \
    MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct \
    HF_TOKEN=hf_xxx \
    python inference.py
"""

import os
import re
import json
import textwrap
from typing import List, Optional
from openai import OpenAI

from client import CustomerServiceEnvClient

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

MAX_STEPS        = 8
TEMPERATURE      = 0.2
MAX_TOKENS       = 200
FALLBACK_ACTION  = "ASK_FOR_MORE_INFO"

VALID_ACTIONS = [
    "GET_ORDER_STATUS",
    "GET_USER_HISTORY",
    "CHECK_REFUND_POLICY",
    "ISSUE_REFUND",
    "DENY_REQUEST",
    "ESCALATE_TO_HUMAN",
    "ASK_FOR_MORE_INFO",
]

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer service agent for an e-commerce company.
    Your goal is to resolve customer queries efficiently and correctly.

    Available actions (reply with EXACTLY one):
    - GET_ORDER_STATUS       → check order delivery status
    - GET_USER_HISTORY       → check customer's past orders and disputes
    - CHECK_REFUND_POLICY    → look up refund eligibility rules
    - ISSUE_REFUND           → approve and process a refund
    - DENY_REQUEST           → reject the customer request with reason
    - ESCALATE_TO_HUMAN      → transfer to human agent
    - ASK_FOR_MORE_INFO      → request missing details from customer

    Rules:
    - Always investigate before making a final decision
    - Check user history if fraud is possible
    - Do NOT issue refunds to fraudulent customers
    - Be efficient — unnecessary actions reduce your score
    - Reply with ONLY the action name, nothing else

    Example reply: GET_ORDER_STATUS
""").strip()


def build_user_prompt(
    step: int,
    obs: dict,
    history: List[str],
) -> str:
    customer_msg  = obs.get("customer_message", "")
    last_result   = obs.get("last_action_result") or "None"
    available     = obs.get("available_actions", VALID_ACTIONS)
    patience      = obs.get("metadata", {}).get("patience_level", "?")
    difficulty    = obs.get("metadata", {}).get("task_difficulty", "?")

    history_text = "\n".join(history[-4:]) if history else "None"

    return textwrap.dedent(f"""
        Step: {step}
        Difficulty: {difficulty}
        Customer patience: {patience}

        Customer message:
        {customer_msg}

        Last action result:
        {last_result}

        Previous steps:
        {history_text}

        Available actions:
        {chr(10).join("- " + a for a in available)}

        Reply with exactly one action name.
    """).strip()


# ── Action Parser ─────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    # Clean up response
    text = response_text.strip().upper()

    # Direct match first
    for action in VALID_ACTIONS:
        if action in text:
            return action

    # Fallback — search line by line
    for line in text.splitlines():
        line = line.strip()
        for action in VALID_ACTIONS:
            if action in line:
                return action

    return FALLBACK_ACTION


# ── Single Episode ────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    env: CustomerServiceEnvClient,
    difficulty: str,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Episode | Difficulty: {difficulty.upper()}")
    print(f"{'='*60}")

    obs      = env.reset(difficulty=difficulty)
    history  : List[str] = []
    done     = False
    total_reward = 0.0

    print(f"Customer: {obs['customer_message']}\n")

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        # Build prompt
        user_prompt = build_user_prompt(step, obs, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        # Call LLM
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM ERROR] {exc} — using fallback action")
            response_text = FALLBACK_ACTION

        action = parse_action(response_text)
        print(f"  Step {step}: {action}")

        # Execute action
        obs          = env.step(action_type=action)
        reward       = obs.get("reward", 0.0)
        done         = obs.get("done", False)
        total_reward += reward

        result = obs.get("last_action_result", "")
        print(f"    Result : {result}")
        print(f"    Reward : {reward:+.2f} | Done: {done}")

        history.append(f"Step {step}: {action} → reward {reward:+.2f}")

    # Grade the episode
    grade_result = env.grade()
    score        = grade_result.get("score", 0.0)
    resolution   = grade_result.get("state_summary", {}).get("resolution_type", "none")

    print(f"\n  Resolution : {resolution}")
    print(f"  Grade Score: {score:.4f}")
    print(f"{'='*60}\n")

    return {
        "difficulty":    difficulty,
        "score":         score,
        "total_reward":  round(total_reward, 4),
        "resolution":    resolution,
        "steps":         obs.get("step_count", 0),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n Customer Service RL Environment — Inference Script")
    print(f" Model     : {MODEL_NAME}")
    print(f" Env URL   : {ENV_BASE_URL}")
    print(f" API URL   : {API_BASE_URL}\n")

    # Init OpenAI client
    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    # Init env client
    env = CustomerServiceEnvClient(base_url=ENV_BASE_URL)

    # Check env is alive
    if not env.health():
        print("ERROR: Environment server not reachable.")
        print(f"Make sure the server is running at {ENV_BASE_URL}")
        print("Run: uvicorn server.app:app --host 0.0.0.0 --port 8000")
        return

    print("Environment server is live.\n")

    # Run all 3 tasks
    results = []
    for difficulty in ["easy", "medium", "hard"]:
        result = run_episode(llm_client, env, difficulty)
        results.append(result)

    # Final summary
    print("\n" + "="*60)
    print("  BENCHMARK RESULTS")
    print("="*60)
    print(f"  {'Task':<10} {'Score':>8} {'Steps':>7} {'Resolution'}")
    print(f"  {'-'*48}")
    for r in results:
        print(
            f"  {r['difficulty']:<10} "
            f"{r['score']:>8.4f} "
            f"{r['steps']:>7} "
            f"{r['resolution']}"
        )

    avg_score = sum(r["score"] for r in results) / len(results)
    print(f"  {'-'*48}")
    print(f"  {'AVERAGE':<10} {avg_score:>8.4f}")
    print("="*60 + "\n")

    env.close()


if __name__ == "__main__":
    main()