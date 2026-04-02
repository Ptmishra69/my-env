import uuid
import random
from typing import Optional, Tuple
from server.scenarios import get_scenario


class CustomerServiceEnvironment:

    MAX_STEPS = 10

    ACTION_COSTS = {
        "GET_ORDER_STATUS":    -0.05,
        "GET_USER_HISTORY":    -0.05,
        "CHECK_REFUND_POLICY": -0.05,
        "ASK_FOR_MORE_INFO":   -0.05,
        "ISSUE_REFUND":         0.0,
        "DENY_REQUEST":         0.0,
        "ESCALATE_TO_HUMAN":   -0.2,
    }

    def __init__(self):
        self._state = None
        self._scenario = None
        self._episode_id = None

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, difficulty: str = "easy") -> dict:
        self._episode_id = str(uuid.uuid4())[:8]
        self._scenario = get_scenario(difficulty)

        self._state = {
            "episode_id":       self._episode_id,
            "step_count":       0,
            "task_difficulty":  difficulty,
            "order_status":     self._scenario["order_status"],
            "refund_eligible":  self._scenario["refund_eligible"],
            "fraud_score":      self._scenario["fraud_score"],
            "customer_patience": self._scenario["customer_patience"],
            "issue_resolved":   False,
            "resolution_type":  None,
            "actions_taken":    [],
            "total_reward":     0.0,
            "order_id_revealed": self._scenario["order_id"] is not None,
            "info_gathered":    False,
        }

        return self._build_observation(
            last_action_result=None,
            reward=0.0,
            done=False,
        )

    def step(self, action_type: str) -> dict:
        if self._state is None:
            raise RuntimeError("Call reset() before step()")

        self._state["step_count"] += 1
        self._state["actions_taken"].append(action_type)

        result, reward, done = self._process_action(action_type)

        # Patience decays each step
        self._state["customer_patience"] = max(
            0.0, self._state["customer_patience"] - 0.05
        )

        # Apply action cost
        reward += self.ACTION_COSTS.get(action_type, 0.0)
        self._state["total_reward"] += reward

        # Force done at max steps
        if self._state["step_count"] >= self.MAX_STEPS:
            done = True

        return self._build_observation(
            last_action_result=result,
            reward=round(reward, 4),
            done=done,
        )

    @property
    def state(self) -> dict:
        if self._state is None:
            return {}
        return self._state.copy()

    # ── Action Processing ─────────────────────────────────────────────────────

    def _process_action(self, action_type: str) -> Tuple[str, float, bool]:
        s = self._state
        sc = self._scenario

        if action_type == "GET_ORDER_STATUS":
            status = sc["order_status"]
            oid = sc.get("order_id", "UNKNOWN") or "NOT PROVIDED"
            result = f"Order {oid}: status = {status.upper()}"
            if status == "delivered":
                result += ". Tracking confirms delivery."
            elif status == "delayed":
                result += ". Estimated delivery: 3-5 more days."
            elif status == "lost":
                result += ". Package marked as lost in transit."
            return result, 0.0, False

        elif action_type == "GET_USER_HISTORY":
            result = f"User history: {sc['user_history']}"
            fraud = sc["fraud_score"]
            if fraud > 0.7:
                result += f" | FRAUD ALERT: risk score {fraud:.2f}"
            return result, 0.0, False

        elif action_type == "CHECK_REFUND_POLICY":
            result = f"Refund policy: {sc['refund_policy']}"
            s["info_gathered"] = True
            return result, 0.0, False

        elif action_type == "ASK_FOR_MORE_INFO":
            if not s["order_id_revealed"] and sc["order_id"] is None:
                s["order_id_revealed"] = True
                sc["order_id"] = f"ORD{random.randint(10000,99999)}"
                result = f"Customer provided order ID: {sc['order_id']}"
                return result, 0.05, False
            else:
                result = "Customer has already provided all needed information."
                return result, -0.1, False

        elif action_type == "ISSUE_REFUND":
            if sc["refund_eligible"] and sc["fraud_score"] < 0.5:
                s["issue_resolved"] = True
                s["resolution_type"] = "refund"
                reward = 1.0
                # Efficiency bonus
                steps = s["step_count"]
                if steps <= 3:
                    reward += 0.2
                elif steps <= 5:
                    reward += 0.1
                return "Refund issued successfully. Customer notified.", reward, True
            elif sc["fraud_score"] >= 0.5:
                s["issue_resolved"] = False
                s["resolution_type"] = "wrong_refund_fraud"
                return "Refund issued. WARNING: High fraud risk ignored.", -1.0, True
            else:
                s["issue_resolved"] = False
                s["resolution_type"] = "wrong_refund"
                return "Refund issued. WARNING: Order was not eligible.", -0.8, True

        elif action_type == "DENY_REQUEST":
            if sc["fraud_score"] >= 0.7:
                s["issue_resolved"] = True
                s["resolution_type"] = "denied_fraud"
                reward = 0.8
                if s["info_gathered"]:
                    reward += 0.1
                return "Request denied. Fraud indicators detected. Case flagged.", reward, True
            elif not sc["refund_eligible"]:
                s["issue_resolved"] = True
                s["resolution_type"] = "denied_policy"
                return "Request denied per policy. Customer informed.", 0.6, True
            else:
                s["issue_resolved"] = False
                s["resolution_type"] = "wrong_denial"
                return "Request denied. WARNING: Customer was eligible for refund.", -0.7, True

        elif action_type == "ESCALATE_TO_HUMAN":
            s["issue_resolved"] = True
            s["resolution_type"] = "escalated"
            if sc["fraud_score"] >= 0.5 or not sc["refund_eligible"]:
                reward = 0.3
            else:
                reward = 0.1
            return "Escalated to human agent. Case transferred.", reward, True

        else:
            return f"Unknown action: {action_type}", -0.2, False

    # ── Observation Builder ───────────────────────────────────────────────────

    def _build_observation(
        self,
        last_action_result: Optional[str],
        reward: float,
        done: bool,
    ) -> dict:
        s = self._state
        sc = self._scenario

        history = []
        actions = s["actions_taken"]
        for i, act in enumerate(actions[-4:]):
            history.append(f"Step {i+1}: {act}")

        return {
            "customer_message":    sc["customer_message"],
            "last_action_result":  last_action_result,
            "conversation_history": history,
            "step_count":          s["step_count"],
            "done":                done,
            "reward":              reward,
            "available_actions": [
                "GET_ORDER_STATUS",
                "GET_USER_HISTORY",
                "CHECK_REFUND_POLICY",
                "ISSUE_REFUND",
                "DENY_REQUEST",
                "ESCALATE_TO_HUMAN",
                "ASK_FOR_MORE_INFO",
            ],
            "metadata": {
                "episode_id":       s["episode_id"],
                "task_difficulty":  s["task_difficulty"],
                "patience_level":   round(s["customer_patience"], 2),
            },
        }