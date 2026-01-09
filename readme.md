A full-stack chess AI project that fine-tunes a pretrained language model on **Viswanathan Anand’s games**, treating chess as a **sequence prediction problem**.  
The model learns to generate plausible next moves in **UCI format**, inspired by Anand’s playing style.


## 🧠 Approach

This project **does NOT use traditional chess engines** or search algorithms.

Instead, it follows a **language-model-based approach**:

- Chess games are converted into **text sequences**:







## 🧩 Model Details

- **Base Model**: `lazy-guy12/chess-llama`
- **Training Data**: PGN games where **Viswanathan Anand** played
- **Objective**: Causal Language Modeling (next-token prediction)
- **Frameworks**:
- PyTorch
- HuggingFace Transformers
- python-chess





## ⚙️ Backend Setup

### 1️⃣ Create virtual environment
```bash
python3 -m venv chess_env
source chess_env/bin/activate


2️⃣ Install dependencies

pip install -r backend/requirements.txt

3️⃣ Run backend server



uvicorn server:app --reload --host 0.0.0.0 --port 8000



🌐 Frontend Setup

cd frontend
npm install
npm run dev


🏋️ Model Training (Notebook)

Training logic is available in:

vishy_trans_trained.ipynb


Val Loss: 1.5915


some other visualizations and results

Loaded 500 positions from 13 games
Evaluating position 500/500...
============================================================
VISHY ANAND IMITATION METRICS
============================================================
Total Positions Evaluated: 500

Exact Match Rate: 48.80%
  → Model picks Vishy's exact move

Top-K Accuracy:
  Top-1: 48.80%
  Top-3: 76.20%
  Top-5: 85.80%
  Top-10: 94.80%
  → Vishy's move appears in model's top-K predictions

Average Rank: 3.03
  → Mean position of Vishy's move in model's ranking

Mean Reciprocal Rank (MRR): 0.6469
  → Harmonic mean of ranks (higher = better)

