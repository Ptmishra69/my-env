import random

SCENARIOS = {
    "easy": [
        {
            "customer_message": "Hi, I received my order #12345 but the item was damaged. I want a refund.",
            "order_status": "delivered",
            "refund_eligible": True,
            "fraud_score": 0.05,
            "customer_patience": 0.9,
            "order_id": "12345",
            "user_history": "2 previous orders, no disputes",
            "refund_policy": "Damaged items are eligible for full refund within 30 days.",
        },
        {
            "customer_message": "My order #99001 arrived broken. Can I get my money back?",
            "order_status": "delivered",
            "refund_eligible": True,
            "fraud_score": 0.03,
            "customer_patience": 0.85,
            "order_id": "99001",
            "user_history": "1 previous order, no disputes",
            "refund_policy": "Damaged items are eligible for full refund within 30 days.",
        },
    ],
    "medium": [
        {
            "customer_message": "My package never arrived and it has been 2 weeks.",
            "order_status": "delayed",
            "refund_eligible": True,
            "fraud_score": 0.2,
            "customer_patience": 0.5,
            "order_id": None,
            "user_history": "3 previous orders, 1 minor dispute resolved",
            "refund_policy": "Orders delayed beyond 10 business days qualify for refund.",
        },
        {
            "customer_message": "I have been waiting for my order forever. Where is it?",
            "order_status": "delayed",
            "refund_eligible": True,
            "fraud_score": 0.15,
            "customer_patience": 0.4,
            "order_id": None,
            "user_history": "5 previous orders, no disputes",
            "refund_policy": "Orders delayed beyond 10 business days qualify for refund.",
        },
    ],
    "hard": [
        {
            "customer_message": "I never received my order #55678. I want a full refund immediately.",
            "order_status": "delivered",
            "refund_eligible": False,
            "fraud_score": 0.85,
            "customer_patience": 0.3,
            "order_id": "55678",
            "user_history": "8 previous refund requests in 6 months, 3 disputed",
            "refund_policy": "Non-delivery claims require investigation if tracking shows delivered.",
        },
        {
            "customer_message": "Order #77412 was not delivered. I demand a refund now.",
            "order_status": "delivered",
            "refund_eligible": False,
            "fraud_score": 0.91,
            "customer_patience": 0.2,
            "order_id": "77412",
            "user_history": "12 refund requests in 1 year, flagged account",
            "refund_policy": "Non-delivery claims require investigation if tracking shows delivered.",
        },
    ],
}


def get_scenario(difficulty: str) -> dict:
    pool = SCENARIOS.get(difficulty, SCENARIOS["easy"])
    scenario = random.choice(pool).copy()
    scenario["difficulty"] = difficulty
    return scenario