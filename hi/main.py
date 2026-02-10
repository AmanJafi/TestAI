import os
import random
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Configure Groq API
api_key = os.getenv("GROQ_API_KEY")
if not api_key or "YOUR" in api_key:
    print("WARNING: GROQ_API_KEY is not set in hi/.env")
    client = None
else:
    client = Groq(api_key=api_key)

app = FastAPI()

# Enable CORS for frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ GAME DATA ------------------

WORDS = {
    "Moth": [
        "Describe its relationship with light.",
        "Describe what happens when it appears uninvited.",
        "Describe how it behaves without intention.",
        "Describe the kind of places it is drawn to.",
        "Describe what it ignores while pursuing something else."
    ],
    "Penguin": [
        "Describe how it moves compared to its environment.",
        "Describe its relationship with temperature.",
        "Describe how it survives without flight.",
        "Describe its social behavior.",
        "Describe how it navigates land versus elsewhere."
    ],
    "Parabola": [
        "Describe its role in motion.",
        "Describe where it naturally appears.",
        "Describe how it balances symmetry and change.",
        "Describe what happens at its most extreme point.",
        "Describe how it redirects paths."
    ],
    "Afghanistan": [
        "Describe its geography through resistance.",
        "Describe the Taliban rule there",
        "Describe how outsiders interact with it.",
        "Describe its oppression of woman.",
        "Describe its relationship with borders."
    ]
}

EASY_PROMPT = """
You are a friendly forest sprite.
Describe the word using clear metaphors and familiar clues.
Be whimsical and helpful.
Never say the word itself.
Max 3 sentences.
"""

MEDIUM_PROMPT = """
You are an abstract philosopher.
Describe the word's essence rather than its function.
Avoid shape, color, or direct usage.
Never say the word itself.
Max 3 sentences.
"""

HARD_PROMPT = """
You describe the given word as an abstract, indirect riddle.

Rules:
•  Never say the word itself
•  Never define it
•  Avoid common associations
•  Speak through absence, consequence, or implication
•  Write as if hiding the answer from an intelligent adversary
•  Use at most 2 sentences
"""

def prompt_for_question(n):
    if n <= 4:
        return EASY_PROMPT
    elif n <= 8:
        return MEDIUM_PROMPT
    else:
        return HARD_PROMPT

# ------------------ GAME STATE ------------------

GAME = {
    "word": None,
    "hints": [],
    "hint_index": 0,
    "question_index": 0,
    "messages": [],
    "finished": False
}

def call_model(messages):
    if not client:
        return "ERROR: Groq API Key not configured."
    
    # The user provided openai/gpt-oss-120b, but we are using Groq.
    # We'll use llama-3.3-70b-versatile for high quality.
    res = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.9,
        top_p=0.95,
        max_tokens=200,
    )
    return res.choices[0].message.content.strip()

# ------------------ ROUTES ------------------

@app.get("/")
async def get_ui():
    ui_path = os.path.join(os.path.dirname(__file__), "..", "chatbot-ui.html")
    return FileResponse(ui_path)

class GuessRequest(BaseModel):
    guess: str

@app.post("/start")
async def start_game():
    word = random.choice(list(WORDS.keys()))
    all_hints = WORDS[word]
    selected_hints = random.sample(all_hints, min(len(all_hints), random.randint(2, 3)))

    GAME["word"] = word
    GAME["hints"] = selected_hints
    GAME["hint_index"] = 0
    GAME["question_index"] = 1
    GAME["finished"] = False

    system_prompt = prompt_for_question(1)

    GAME["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Word: {word}"}
    ]

    clue = call_model(GAME["messages"])
    GAME["messages"].append({"role": "assistant", "content": clue})

    return {
        "question": 1,
        "difficulty": "easy",
        "text": clue
    }

@app.post("/next")
async def next_hint():
    if GAME["finished"]:
        return {"error": "Game over", "status_code": 400}

    GAME["question_index"] += 1

    if GAME["question_index"] > 10:
        GAME["finished"] = True
        return {"message": "No more hints", "finished": True}

    system_prompt = prompt_for_question(GAME["question_index"])
    GAME["messages"][0] = {"role": "system", "content": system_prompt}

    if GAME["hint_index"] < len(GAME["hints"]):
        hint_prompt = GAME["hints"][GAME["hint_index"]]
        GAME["hint_index"] += 1
        user_msg = f"Answer this hint indirectly: {hint_prompt}"
    else:
        user_msg = "Give another indirect clue. Do not repeat previous clues."

    GAME["messages"].append({"role": "user", "content": user_msg})

    reply = call_model(GAME["messages"])
    GAME["messages"].append({"role": "assistant", "content": reply})

    difficulty = (
        "easy" if GAME["question_index"] <= 4
        else "medium" if GAME["question_index"] <= 8
        else "hard"
    )

    return {
        "question": GAME["question_index"],
        "difficulty": difficulty,
        "text": reply
    }

@app.post("/guess")
async def guess(request: GuessRequest):
    if GAME["word"] is None:
         return {"correct": False, "message": "Game not started yet."}
         
    user_guess = request.guess.strip().lower()
    answer = GAME["word"].lower()

    if user_guess == answer:
        GAME["finished"] = True
        return {
            "correct": True,
            "message": f"Correct! The word was {GAME['word']}!"
        }

    return {
        "correct": False,
        "message": "Wrong guess. Try again or ask for another hint!"
    }

@app.get("/state")
async def get_state():
    return {
        "question": GAME["question_index"],
        "finished": GAME["finished"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)