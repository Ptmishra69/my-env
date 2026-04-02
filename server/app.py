from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import json

from server.environment import CustomerServiceEnvironment
from server.graders import run_grader

app = FastAPI(
    title="Customer Service RL Environment",
    description="OpenEnv-compatible customer service simulation",
    version="1.0.0",
)

# One environment instance per WebSocket session
# For HTTP endpoints we use a shared instance
_env = CustomerServiceEnvironment()


# ── Request Models ────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = "easy"

class StepRequest(BaseModel):
    action_type: str
    metadata: Optional[dict] = {}


# ── HTTP Endpoints ────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/reset")
async def reset(request: ResetRequest = ResetRequest()):
    """
    Reset the environment and start a new episode.
    Accepts empty body {} — difficulty defaults to 'easy'.
    """
    difficulty = request.difficulty or "easy"
    obs = _env.reset(difficulty=difficulty)
    return JSONResponse(content=obs)


@app.post("/step")
async def step(request: StepRequest):
    """Execute one action and return the new observation."""
    try:
        obs = _env.step(action_type=request.action_type)
        return JSONResponse(content=obs)
    except RuntimeError as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e), "hint": "Call /reset first"}
        )


@app.get("/state")
async def get_state():
    """Return the full hidden state (for debugging/judging)."""
    return JSONResponse(content=_env.state)


@app.get("/grade")
async def grade():
    """Run the grader on the current episode state."""
    state = _env.state
    difficulty = state.get("task_difficulty", "easy")
    score = run_grader(difficulty, state)
    return JSONResponse(content={
        "difficulty": difficulty,
        "score": score,
        "state_summary": {
            "resolution_type": state.get("resolution_type"),
            "steps_taken": state.get("step_count"),
            "issue_resolved": state.get("issue_resolved"),
        }
    })


# ── WebSocket Endpoint ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Persistent WebSocket session — one environment instance per connection.
    Messages: {"type": "reset", "difficulty": "easy"} or
              {"type": "step", "action_type": "GET_ORDER_STATUS"}
    """
    await websocket.accept()
    session_env = CustomerServiceEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            msg_type = data.get("type")

            if msg_type == "reset":
                difficulty = data.get("difficulty", "easy")
                obs = session_env.reset(difficulty=difficulty)
                await websocket.send_json({"type": "observation", "data": obs})

            elif msg_type == "step":
                action_type = data.get("action_type", "")
                try:
                    obs = session_env.step(action_type=action_type)
                    await websocket.send_json({"type": "observation", "data": obs})
                except RuntimeError as e:
                    await websocket.send_json({"type": "error", "message": str(e)})

            elif msg_type == "state":
                await websocket.send_json({
                    "type": "state",
                    "data": session_env.state
                })

            elif msg_type == "grade":
                state = session_env.state
                difficulty = state.get("task_difficulty", "easy")
                score = run_grader(difficulty, state)
                await websocket.send_json({
                    "type": "grade",
                    "score": score,
                    "difficulty": difficulty,
                })

            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass

# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
    )

if __name__ == "__main__":
    main()