from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import chess
from transformers import AutoTokenizer, AutoModelForCausalLM

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

# ===================== MODEL =====================
MODEL_ID = "lazy-guy12/chess-llama"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ===================== API SCHEMA =====================
class Position(BaseModel):
    moves: list[str]   # list of UCI moves
    difficulty: int | None = 0  # 0 = best, higher = weaker

# ===================== POLICY MOVE SELECTION =====================
def select_move_policy(board: chess.Board, moves: list[str], difficulty: int = 0):
    """
    Implements EXACT Chess-LLaMA logic:
    - forward pass only
    - logits for NEXT token
    - rank moves by probability
    - pick first legal move
    """

    # Build input string (same as llama.ts)
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

    # logits for NEXT token only
    next_logits = outputs.logits[:, -1, :]

    # probabilities
    probs = torch.softmax(next_logits, dim=-1)

    # sort by probability (descending)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)

    sorted_ids = sorted_ids[0]  # batch size = 1

    # build legal destinations map (like dests in TS)
    legal_moves = {m.uci() for m in board.legal_moves}

    # difficulty offset (same idea as `mid`)
    start_index = difficulty

    # search strongest → weaker
    for i in range(start_index, len(sorted_ids)):
        token_id = sorted_ids[i].item()
        move = tokenizer.decode([token_id])

        if len(move) < 4:
            continue

        move = move[:4]

        if move in legal_moves:
            print(f"✅ Selected move: {move} (rank {i})")
            return move

    # fallback (should be extremely rare)
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
        "model": MODEL_ID,
        "device": device
    }
