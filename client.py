import requests
import json
from typing import Optional


class CustomerServiceEnvClient:
    """
    HTTP client for the Customer Service RL Environment.
    Wraps the FastAPI server endpoints with a clean Python API.
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def reset(self, difficulty: str = "easy") -> dict:
        """Start a new episode. Returns the first observation."""
        resp = self.session.post(
            f"{self.base_url}/reset",
            json={"difficulty": difficulty},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str) -> dict:
        """Execute one action. Returns observation + reward + done."""
        resp = self.session.post(
            f"{self.base_url}/step",
            json={"action_type": action_type},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()

    def state(self) -> dict:
        """Get the full hidden state (for debugging)."""
        resp = self.session.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def grade(self) -> dict:
        """Run grader on current episode. Returns score in [0.0, 1.0]."""
        resp = self.session.get(f"{self.base_url}/grade", timeout=30)
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Check if server is alive."""
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=10)
            return resp.status_code == 200
        except Exception:
            return False

    def close(self):
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ── Quick test when run directly ─────────────────────────────────────────────

if __name__ == "__main__":
    client = CustomerServiceEnvClient()

    print("Testing connection...")
    print(f"Health: {client.health()}")

    print("\n--- Easy episode ---")
    obs = client.reset(difficulty="easy")
    print(f"Customer: {obs['customer_message']}")

    for action in ["GET_ORDER_STATUS", "CHECK_REFUND_POLICY", "ISSUE_REFUND"]:
        obs = client.step(action)
        print(f"Action: {action}")
        print(f"Result: {obs['last_action_result']}")
        print(f"Reward: {obs['reward']} | Done: {obs['done']}")
        if obs["done"]:
            break

    result = client.grade()
    print(f"\nFinal score: {result['score']}")
    client.close()