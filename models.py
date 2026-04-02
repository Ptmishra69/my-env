from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


# ── Actions ──────────────────────────────────────────────────────────────────

@dataclass
class CustomerServiceAction:
    action_type: str          # one of the 7 valid actions below
    metadata: Dict[str, Any] = field(default_factory=dict)

VALID_ACTIONS = [
    "GET_ORDER_STATUS",
    "GET_USER_HISTORY",
    "CHECK_REFUND_POLICY",
    "ISSUE_REFUND",
    "DENY_REQUEST",
    "ESCALATE_TO_HUMAN",
    "ASK_FOR_MORE_INFO",
]


# ── Observation ───────────────────────────────────────────────────────────────

@dataclass
class CustomerServiceObservation:
    customer_message: str
    last_action_result: Optional[str]
    conversation_history: List[str]
    step_count: int
    done: bool
    reward: Optional[float]
    available_actions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── State (hidden truth — never sent to agent) ────────────────────────────────

@dataclass
class CustomerServiceState:
    episode_id: str
    step_count: int
    task_difficulty: str        # "easy" | "medium" | "hard"
    order_status: str           # "delivered" | "delayed" | "lost"
    refund_eligible: bool
    fraud_score: float          # 0.0 – 1.0
    customer_patience: float    # 1.0 = calm, 0.0 = very angry
    issue_resolved: bool
    resolution_type: Optional[str]   # "refund" | "denied" | "escalated" | None
    actions_taken: List[str]
    total_reward: float