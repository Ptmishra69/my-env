"""
Inference Script — Customer Service RL Environment
===================================================
MANDATORY ENV VARS:
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
import textwrap
from typing import List, Optional
from openai import OpenAI

from client import CustomerServiceEnvClient

# ── Config ────────────────────────────────────────────────────────────────────

# ✅ These are injected by validator — use os.environ
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]

# ✅ This may NOT be injected — keep a safe default
MODEL_NAME       = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
ENV_BASE_URL     = os.getenv("ENV_BASE_URL", "http://localhost:8000")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

MAX_STEPS       = 8
TEMPERATURE     = 0.1
MAX_TOKENS      = 50
FALLBACK_ACTION = "ESCALATE_TO_HUMAN"

VALID_ACTIONS = [
    "GET_ORDER_STATUS",
    "GET_USER_HISTORY",
    "CHECK_REFUND_POLICY",
    "ISSUE_REFUND",
    "DENY_REQUEST",
    "ESCALATE_TO_HUMAN",
    "ASK_FOR_MORE_INFO",
]

INFO_ACTIONS = {
    "GET_ORDER_STATUS",
    "GET_USER_HISTORY",
    "CHECK_REFUND_POLICY",
    "ASK_FOR_MORE_INFO",
}

TERMINAL_ACTIONS = {
    "ISSUE_REFUND",
    "DENY_REQUEST",
    "ESCALATE_TO_HUMAN",
}

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are an AI customer service agent for an e-commerce company.
    Resolve customer queries in as FEW steps as possible — efficiency is rewarded.

    Available actions:
    - GET_ORDER_STATUS       → check order delivery status
    - GET_USER_HISTORY       → check customer's past orders and fraud signals
    - CHECK_REFUND_POLICY    → look up refund eligibility rules
    - ISSUE_REFUND           → approve and process a refund (terminal)
    - DENY_REQUEST           → reject the request (terminal)
    - ESCALATE_TO_HUMAN      → transfer to human agent (terminal)
    - ASK_FOR_MORE_INFO      → request missing details

    DECISION RULES — follow strictly:
    1. If fraud risk is HIGH (fraud score >= 0.7 or many past disputes) → DENY_REQUEST
    2. If order is eligible and customer is legitimate → ISSUE_REFUND
    3. If order is not eligible and no fraud → DENY_REQUEST
    4. If situation is truly unclear → ESCALATE_TO_HUMAN
    5. NEVER repeat the same info-gathering action twice
    6. After gathering enough info (2-3 steps max), you MUST make a final decision

    Reply with ONLY the action name. Nothing else.
""").strip()


def build_user_prompt(step: int, obs: dict, history: List[str], actions_taken: List[str]) -> str:
    customer_msg = obs.get("customer_message", "")
    last_result  = obs.get("last_action_result") or "None"
    available    = obs.get("available_actions", VALID_ACTIONS)
    patience     = obs.get("metadata", {}).get("patience_level", "?")
    difficulty   = obs.get("metadata", {}).get("task_difficulty", "?")
    history_text = "\n".join(history[-6:]) if history else "None"

    steps_remaining = MAX_STEPS - step
    urgency = ""
    if steps_remaining <= 2:
        urgency = f"\n⚠️  URGENT: Only {steps_remaining} steps left — you MUST choose ISSUE_REFUND, DENY_REQUEST, or ESCALATE_TO_HUMAN NOW."
    elif step >= 3:
        urgency = "\nYou have gathered enough information. Make your final decision now."

    already_done = ", ".join(actions_taken) if actions_taken else "None"

    return textwrap.dedent(f"""
        Step {step} of {MAX_STEPS} | Difficulty: {difficulty} | Customer patience: {patience}

        Customer message:
        {customer_msg}

        Last action result:
        {last_result}

        Actions already taken: {already_done}

        Full history:
        {history_text}
        {urgency}

        Available actions:
        {chr(10).join("- " + a for a in available)}

        Reply with exactly one action name.
    """).strip()


# ── Rule-based fallback decision ──────────────────────────────────────────────

def rule_based_decision(history: List[str], actions_taken: List[str]) -> Optional[str]:
    history_text = " ".join(history).upper()

    if "FRAUD" in history_text or "RISK SCORE" in history_text or "FLAGGED" in history_text:
        return "DENY_REQUEST"

    if ("ELIGIBLE" in history_text or "QUALIFY" in history_text) and "FRAUD" not in history_text:
        return "ISSUE_REFUND"

    if "GET_USER_HISTORY" not in actions_taken:
        return "GET_USER_HISTORY"

    if "GET_ORDER_STATUS" not in actions_taken:
        return "GET_ORDER_STATUS"

    info_count = sum(1 for a in actions_taken if a in INFO_ACTIONS)
    if info_count >= 3:
        return "ESCALATE_TO_HUMAN"

    return None


# ── Action Parser ─────────────────────────────────────────────────────────────

def parse_action(response_text: str) -> str:
    if not response_text:
        return FALLBACK_ACTION

    text = response_text.strip().upper()

    for action in VALID_ACTIONS:
        if action in text:
            return action

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
    task_name = f"customer_service_{difficulty}"

    print(f"\n{'='*60}")
    print(f"  Episode | Difficulty: {difficulty.upper()}")
    print(f"{'='*60}")

    print(f"[START] task={task_name}", flush=True)

    obs           = env.reset(difficulty=difficulty)
    history       : List[str] = []
    actions_taken : List[str] = []
    done          = False
    total_reward  = 0.0

    print(f"Customer: {obs['customer_message']}\n")

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        steps_remaining = MAX_STEPS - step
        action = None

        if steps_remaining <= 1 or (step >= 4 and all(a in INFO_ACTIONS for a in actions_taken)):
            action = rule_based_decision(history, actions_taken)
            if action:
                print(f"  Step {step}: {action} [rule-based]")

        if action is None:
            user_prompt = build_user_prompt(step, obs, history, actions_taken)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ]

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
                print(f"  [LLM ERROR] {exc} — using rule-based fallback")
                response_text = ""

            action = parse_action(response_text)

            if step >= 4 and action in INFO_ACTIONS and action in actions_taken:
                loop_break = rule_based_decision(history, actions_taken)
                action = loop_break or "ESCALATE_TO_HUMAN"
                print(f"  Step {step}: {action} [loop-break]")
            else:
                print(f"  Step {step}: {action}")

        actions_taken.append(action)

        obs          = env.step(action_type=action)
        reward       = obs.get("reward", 0.0)
        done         = obs.get("done", False)
        total_reward += reward

        result = obs.get("last_action_result", "")
        print(f"    Result : {result}")
        print(f"    Reward : {reward:+.2f} | Done: {done}")

        history.append(f"Step {step}: {action} → {result} (reward {reward:+.2f})")

        print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)

    grade_result = env.grade()
    score        = grade_result.get("score", 0.0)
    resolution   = grade_result.get("state_summary", {}).get("resolution_type", "none")
    step_count   = obs.get("step_count", 0)

    print(f"\n  Resolution : {resolution}")
    print(f"  Grade Score: {score:.4f}")
    print(f"{'='*60}\n")

    print(f"[END] task={task_name} score={round(score, 4)} steps={step_count}", flush=True)

    return {
        "difficulty":   difficulty,
        "score":        score,
        "total_reward": round(total_reward, 4),
        "resolution":   resolution,
        "steps":        step_count,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n Customer Service RL Environment — Inference Script")
    print(f" Model     : {MODEL_NAME}")
    print(f" Env URL   : {ENV_BASE_URL}")
    print(f" API URL   : {API_BASE_URL}\n")

    llm_client = OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )

    env = CustomerServiceEnvClient(base_url=ENV_BASE_URL)

    if not env.health():
        # Emit required blocks so validator does not fail on missing output
        for difficulty in ["easy", "medium", "hard"]:
            task_name = f"customer_service_{difficulty}"
            print(f"[START] task={task_name}", flush=True)
            print(f"[STEP] step=1 reward=0.0", flush=True)
            print(f"[END] task={task_name} score=0.0 steps=1", flush=True)
        print("ERROR: Environment server not reachable.", flush=True)
        print(f"Make sure the server is running at {ENV_BASE_URL}")
        return

    print("Environment server is live.\n")

    results = []
    for difficulty in ["easy", "medium", "hard"]:
        result = run_episode(llm_client, env, difficulty)
        results.append(result)

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