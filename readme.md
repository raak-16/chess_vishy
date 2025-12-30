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

python backend/server.py



🌐 Frontend Setup

cd frontend
npm install
npm run dev


🏋️ Model Training (Notebook)

Training logic is available in:

vishy_trans_trained.ipynb


