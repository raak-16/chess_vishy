from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import chess
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# ===================== APP =====================
app = FastAPI()

# ===================== CORS =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== DEVICE =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ===================== LOCAL MODEL =====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "vishy_trans")

print("Loading model from:", MODEL_DIR)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    local_files_only=True
)

model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ===================== API SCHEMA =====================
class Position(BaseModel):
    moves: list[str]           # UCI history
    difficulty: int | None = 0 # 0 = strongest (like mid)

# ===================== POLICY MOVE SELECTION =====================
def select_move_policy(board: chess.Board, moves: list[str], difficulty: int = 0):
    """
    EXACT equivalent of llama.ts:
    - forward() only
    - logits[:, -1]
    - rank tokens
    - pick first legal
    """

    # same as llama.ts: moves.join(" ")
    text = " ".join(moves)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # NEXT token logits only
    logits = outputs.logits[:, -1, :]

    # probabilities
    probs = torch.softmax(logits, dim=-1)

    # sort descending
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    sorted_ids = sorted_ids[0]  # batch size = 1

    legal_moves = {m.uci() for m in board.legal_moves}

    # difficulty offset (same as mid)
    start = difficulty if difficulty < len(sorted_ids) else 0

    # search strongest → weaker
    for i in range(start, len(sorted_ids)):
        token_id = sorted_ids[i].item()
        move = tokenizer.decode([token_id])

        if len(move) < 4:
            continue

        move = move[:4]

        if move in legal_moves:
            print(f"✅ Selected move: {move} (rank {i})")
            return move

    # very rare fallback
    fallback = next(iter(legal_moves))
    print(f"⚠️ Fallback move: {fallback}")
    return fallback

# ===================== API ENDPOINT =====================
@app.post("/move")
def get_move(position: Position):
    print("\n" + "=" * 50)
    print("Moves so far:", position.moves)

    board = chess.Board()
    for m in position.moves:
        board.push_uci(m)

    print("Turn:", "White" if board.turn else "Black")

    move = select_move_policy(
        board,
        position.moves,
        position.difficulty or 0
    )

    print("=" * 50)
    return {"move": move}

# ===================== HEALTH =====================
@app.get("/")
def health():
    return {
        "status": "ok",
        "model": "vishy_trans",
        "device": device
    }
