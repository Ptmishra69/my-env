"""
Graders for each task difficulty.
Each grader returns a float in [0.0, 1.0] with partial credit.
"""


def grade_easy(state: dict) -> float:
    """
    Easy task: Customer has damaged item, clear refund eligibility.
    Agent should: GET_ORDER_STATUS → ISSUE_REFUND (in ≤ 4 steps)
    """
    if not state:
        return 0.0

    resolution = state.get("resolution_type")
    steps = state.get("step_count", 10)
    actions = state.get("actions_taken", [])

    # Wrong resolution
    if resolution in ("wrong_refund", "wrong_denial", "wrong_refund_fraud"):
        return 0.0

    # Correct resolution: refund issued
    if resolution == "refund":
        score = 1.0
        # Efficiency bonus: fewer steps = better
        if steps <= 2:
            score = 1.0
        elif steps <= 4:
            score = 0.85
        elif steps <= 6:
            score = 0.65
        else:
            score = 0.45
        return round(score, 2)

    # Escalated instead of resolving (partial credit)
    if resolution == "escalated":
        return 0.3

    # Did useful actions but didn't resolve
    useful = {"GET_ORDER_STATUS", "CHECK_REFUND_POLICY"}
    useful_taken = len([a for a in actions if a in useful])
    if useful_taken >= 1:
        return 0.2

    return 0.0


def grade_medium(state: dict) -> float:
    """
    Medium task: Delayed order, missing order ID.
    Agent should: ASK_FOR_MORE_INFO → GET_ORDER_STATUS → ISSUE_REFUND
    """
    if not state:
        return 0.0

    resolution = state.get("resolution_type")
    steps = state.get("step_count", 10)
    actions = state.get("actions_taken", [])

    asked_info = "ASK_FOR_MORE_INFO" in actions
    checked_status = "GET_ORDER_STATUS" in actions

    # Wrong resolution
    if resolution in ("wrong_refund", "wrong_denial", "wrong_refund_fraud"):
        return 0.05 if asked_info else 0.0

    # Correct refund issued
    if resolution == "refund":
        score = 1.0
        if not asked_info:
            score -= 0.2      # Penalise skipping info gathering
        if not checked_status:
            score -= 0.15
        if steps > 6:
            score -= 0.1 * (steps - 6) / 4
        return round(max(0.1, score), 2)

    # Denied a delayed order (wrong)
    if resolution in ("denied_policy", "denied_fraud"):
        return 0.1

    # Escalated
    if resolution == "escalated":
        base = 0.35
        if asked_info:
            base += 0.1
        return round(base, 2)

    # Partial: gathered info but didn't resolve
    partial = 0.0
    if asked_info:
        partial += 0.15
    if checked_status:
        partial += 0.1
    if "CHECK_REFUND_POLICY" in actions:
        partial += 0.05

    return round(min(partial, 0.35), 2)


def grade_hard(state: dict) -> float:
    """
    Hard task: Fraud detected, tracking shows delivered.
    Agent should: GET_USER_HISTORY → GET_ORDER_STATUS → DENY_REQUEST
    """
    if not state:
        return 0.0

    resolution = state.get("resolution_type")
    steps = state.get("step_count", 10)
    actions = state.get("actions_taken", [])

    checked_history = "GET_USER_HISTORY" in actions
    checked_status = "GET_ORDER_STATUS" in actions
    checked_policy = "CHECK_REFUND_POLICY" in actions

    investigation_score = 0.0
    if checked_history:
        investigation_score += 0.15
    if checked_status:
        investigation_score += 0.10
    if checked_policy:
        investigation_score += 0.05

    # Issued refund to fraud customer (worst outcome)
    if resolution in ("wrong_refund_fraud", "wrong_refund"):
        return round(min(0.05, investigation_score * 0.2), 2)

    # Denied correctly (fraud detected)
    if resolution in ("denied_fraud", "denied_policy"):
        score = 0.7
        score += investigation_score
        if steps <= 4:
            score += 0.1
        elif steps <= 6:
            score += 0.05
        return round(min(score, 1.0), 2)

    # Escalated (acceptable but not ideal for fraud)
    if resolution == "escalated":
        base = 0.4
        base += investigation_score
        return round(min(base, 0.65), 2)

    # Wrong denial (eligible customer denied — not the case here but guard)
    if resolution == "wrong_denial":
        return round(investigation_score * 0.3, 2)

    # No resolution yet: partial investigation credit
    return round(min(investigation_score, 0.3), 2)


def run_grader(difficulty: str, state: dict) -> float:
    """Entry point — routes to the correct grader."""
    graders = {
        "easy":   grade_easy,
        "medium": grade_medium,
        "hard":   grade_hard,
    }
    grader = graders.get(difficulty, grade_easy)
    score = grader(state)
    return round(max(0.0, min(1.0, score)), 4)