---
title: Customer Service RL Environment
emoji: 🎧
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Customer Service RL Environment

An OpenEnv-compatible reinforcement learning environment that simulates
a real-world e-commerce customer service system.

## What the Agent Does

An AI agent receives customer queries and must resolve them by calling
tools — checking order status, user history, refund policies — and
making final decisions (issue refund, deny, escalate).

## Action Space

| Action | Description |
|---|---|
| GET_ORDER_STATUS | Check delivery status of the order |
| GET_USER_HISTORY | Check customer's past orders and fraud signals |
| CHECK_REFUND_POLICY | Look up eligibility rules |
| ISSUE_REFUND | Approve and process a refund |
| DENY_REQUEST | Reject the request |
| ESCALATE_TO_HUMAN | Transfer to human agent |
| ASK_FOR_MORE_INFO | Request missing details from customer |

## Observation Space

| Field | Type | Description |
|---|---|---|
| customer_message | string | The customer's complaint |
| last_action_result | string | Result of the last action taken |
| conversation_history | list | Last 4 steps taken |
| step_count | int | Current step number |
| done | bool | Whether episode is complete |
| reward | float | Reward from last action |
| available_actions | list | Valid actions |

## 3 Tasks

- **Easy** — Damaged item, clear refund eligibility
- **Medium** — Delayed order, missing order ID, requires info gathering
- **Hard** — Fraud detection, tracking shows delivered, must deny

## Setup
```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| /health | GET | Health check |
| /reset | POST | Start new episode |
| /step | POST | Execute action |
| /state | GET | Get hidden state |
| /grade | GET | Get episode score |
| /ws | WebSocket | Persistent session |