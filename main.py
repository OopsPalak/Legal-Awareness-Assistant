from uuid import uuid4
from time import perf_counter
from typing import Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import logging

# ------------------------------------------------------------------
# LLM client (Ollama, OpenAI-compatible)
# ------------------------------------------------------------------
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama server
    api_key="ollama",                      # dummy string, NOT a real key
)

# ------------------------------------------------------------------
# FastAPI app + CORS
# ------------------------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# ------------------------------------------------------------------
# In‑memory session + observability data
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    sessionId: str | None = None
    message: str


class SessionOp(BaseModel):
    sessionId: str


# conversation memory: session_id -> list[{"role": "...", "content": "..."}]
conversation_history: Dict[str, List[dict]] = {}

# long‑running operations: paused sessions
session_state: Dict[str, dict] = {}  # e.g. {"paused": True/False}


def get_state(session_id: str) -> dict:
    if session_id not in session_state:
        session_state[session_id] = {"paused": False}
    return session_state[session_id]


# ------------------------------------------------------------------
# Helper: simple agent evaluation for observability
# ------------------------------------------------------------------
def evaluate_answer(answer: str) -> float:
    """
    Very simple heuristic "evaluation":
    +0.5 if it has 'Your basic rights:'
    +0.5 if it has 'What you can do:'
    """
    score = 0.0
    if "Your basic rights:" in answer:
        score += 0.5
    if "What you can do:" in answer:
        score += 0.5
    return score


# ------------------------------------------------------------------
# Agent 1: Safety agent (LLM-powered, part of multi-agent system)
# ------------------------------------------------------------------
def run_safety_agent(user_text: str) -> str:
    safety_system_prompt = (
        "You are a safety checker for a legal awareness chatbot.\n"
        "User messages may describe crime, danger, self-harm or abuse.\n"
        "Reply with ONLY ONE short line:\n"
        "- 'OK' if this looks normal.\n"
        "- Or 'ALERT: <very short reason>' if it seems urgent or dangerous."
    )

    resp = client.chat.completions.create(
        model="llama3.2:3b",
        temperature=0.0,
        max_tokens=32,
        messages=[
            {"role": "system", "content": safety_system_prompt},
            {"role": "user", "content": user_text},
        ],
    )
    return resp.choices[0].message.content.strip()


# ------------------------------------------------------------------
# Agent 2: Main legal awareness explainer (LLM-powered)
# ------------------------------------------------------------------
def run_legal_agent(session_id: str, user_text: str, safety_note: str | None) -> str:
    system_prompt = (
        "You are a Legal Awareness Assistant for students and ordinary people.\n\n"
        "ROLE:\n"
        "- You ONLY give general legal information and safety guidance.\n"
        "- You are NOT a lawyer and NOT a substitute for real legal advice.\n"
        "- If a situation sounds urgent, dangerous, or involves crime, always tell "
        "the user to contact local authorities, a trusted adult, or a real lawyer.\n\n"
        "STYLE RULES:\n"
        "- Answer in very simple English.\n"
        "- Use short sentences.\n"
        "- Never write one huge paragraph.\n"
        "- Always structure the answer in 3 parts:\n"
        "  1) A short 1–2 line summary.\n"
        "  2) A section titled 'Your basic rights:' with 3–5 bullet points starting with '- '.\n"
        "  3) A section titled 'What you can do:' with 3–5 bullet points starting with '- '.\n"
        "- Leave a blank line between sections.\n"
        "- Do NOT invent exact law numbers or sections.\n"
    )

    history = conversation_history.get(session_id, [])

    messages: List[dict] = [{"role": "system", "content": system_prompt}]

    # add up to last 8 turns of context for memory
    for m in history[-8:]:
        messages.append(m)

    # add optional safety note so main agent sees it
    if safety_note and safety_note.startswith("ALERT"):
        user_text = (
            user_text
            + "\n\n[Safety agent note: "
            + safety_note
            + " – remember to give calm, safe guidance.]"
        )

    messages.append({"role": "user", "content": user_text})

    resp = client.chat.completions.create(
        model="llama3.2:3b",
        temperature=0.4,
        max_tokens=256,
        messages=messages,
    )
    answer = resp.choices[0].message.content
    # update memory
    history.extend(
        [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]
    )
    conversation_history[session_id] = history
    return answer


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
async def home():
    return {"message": "AI Legal Chatbot is running with Ollama."}


@app.post("/chat")
async def chat_http(body: ChatRequest):
    session_id = body.sessionId or str(uuid4())
    state = get_state(session_id)

    if state.get("paused"):
        # long‑running operations feature: session can be paused
        return {
            "sessionId": session_id,
            "reply": "This conversation is currently paused. Please resume to continue.",
        }

    start = perf_counter()

    # Multi‑agent: safety agent first
    safety_note = run_safety_agent(body.message)

    # Main agent
    reply = run_legal_agent(session_id, body.message, safety_note)

    duration_ms = (perf_counter() - start) * 1000
    score = evaluate_answer(reply)

    # Observability logs
    logging.info(
        {
            "event": "chat_completion",
            "sessionId": session_id,
            "duration_ms": round(duration_ms, 1),
            "safety_note": safety_note,
            "eval_score": score,
        }
    )

    return {
        "sessionId": session_id,
        "reply": reply,
        "meta": {
            "duration_ms": duration_ms,
            "safety_note": safety_note,
            "eval_score": score,
        },
    }


@app.post("/reset-session")
async def reset_session(body: SessionOp):
    # clear memory for that session
    conversation_history.pop(body.sessionId, None)
    state = get_state(body.sessionId)
    state["paused"] = False
    return {"status": "ok"}


@app.post("/pause-session")
async def pause_session(body: SessionOp):
    state = get_state(body.sessionId)
    state["paused"] = True
    return {"status": "paused"}


@app.post("/resume-session")
async def resume_session(body: SessionOp):
    state = get_state(body.sessionId)
    state["paused"] = False
    return {"status": "resumed"}
