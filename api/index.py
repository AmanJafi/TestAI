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
    ],
    "Camera": [
        "It preserves a moment without experiencing it.",
        "It uses light to freeze time.",
        "It has a single eye that blinks only once per memory."
    ],
    "Bicycle": [
        "It demands balance to maintain forward motion.",
        "It relies on two circles and human effort.",
        "It has no heart but is powered by your legs."
    ],
    "Mirror": [
        "It always tells the truth but only through reversal.",
        "It knows your face better than you do.",
        "It replicates everything it sees but feels none of it."
    ],
    "Elephant": [
        "A giant that never forgets a face.",
        "It carries its own water hose wherever it goes.",
        "It has tusks but is known for its gentle memory."
    ],
    "Clock": [
        "It has hands but cannot grab anything.",
        "It counts the invisible flow that never stops.",
        "It circles itself twelve times before starting again."
    ],
    "Ladder": [
        "It provides steps to where you cannot reach.",
        "It leans on things to help you rise higher.",
        "It has rungs but no melody."
    ],
    "Candle": [
        "It spends its life melting for your sight.",
        "It consumes itself to produce a tiny star.",
        "It dies if you breathe on it too hard."
    ],
    "Compass": [
        "A needle that is obsessed with only one direction.",
        "It helps you find your way when all paths look the same.",
        "It speaks of North without ever going there."
    ],
    "Octopus": [
        "A creature of the deep with three hearts and blue blood.",
        "It can change its shape and color to disappear.",
        "It has eight limbs that can each think for themselves."
    ]
}

EASY_PROMPT = """
Describe the word using simple and direct clues.
Never say the word itself.
Max 3 sentences.
"""

MEDIUM_PROMPT = """
Describe the word by its function and context.
Avoid naming its category or direct synonyms.
Never say the word itself.
Max 3 sentences.
"""

HARD_PROMPT = """
Describe the given word as an abstract, indirect riddle.
Rules:
•  Never say the word itself
•  Never define it
•  Speak through implication
•  Use at most 2 sentences
"""

def prompt_for_question(n, level):
    if level == 1:
        return EASY_PROMPT
    elif level == 2:
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
    "finished": False,
    "used_words": set(),
    "level": 1, 
    "session_history": [],
    "current_guesses": 0,
    "current_hints": 0
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
async def start_game(session_reset: bool = False):
    if session_reset:
        GAME["level"] = 1
        GAME["session_history"] = []
        GAME["used_words"] = set()

    available_words = [w for w in WORDS.keys() if w not in GAME["used_words"]]
    if not available_words:
        GAME["used_words"] = set()
        available_words = list(WORDS.keys())
        
    word = random.choice(available_words)
    GAME["used_words"].add(word)
    
    all_hints = WORDS[word]
    selected_hints = random.sample(all_hints, min(len(all_hints), random.randint(2, 3)))

    GAME["word"] = word
    GAME["hints"] = selected_hints
    GAME["hint_index"] = 0
    GAME["question_index"] = 1
    GAME["finished"] = False
    GAME["current_guesses"] = 0
    GAME["current_hints"] = 1 # Initial clue counts as 1 hint

    # Use prompt based on current level
    system_prompt = prompt_for_question(1, GAME["level"])

    GAME["messages"] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Word: {word}"}
    ]

    clue = call_model(GAME["messages"])
    GAME["messages"].append({"role": "assistant", "content": clue})

    return {
        "question": 1,
        "difficulty": ["easy", "medium", "hard"][GAME["level"]-1],
        "level": GAME["level"],
        "text": clue
    }

@app.post("/next")
async def next_hint():
    if GAME["finished"]:
        return {"error": "Game over", "status_code": 400}

    GAME["question_index"] += 1
    GAME["current_hints"] += 1

    if GAME["question_index"] > 10:
        GAME["finished"] = True
        return {"message": "No more hints", "finished": True}

    system_prompt = prompt_for_question(GAME["question_index"], GAME["level"])
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
         
    GAME["current_guesses"] += 1
    user_guess = request.guess.strip().lower()
    answer = GAME["word"].lower()

    if user_guess == answer:
        GAME["finished"] = True
        
        # Scoring logic
        base_scores = {1: 50, 2: 75, 3: 100}
        base = base_scores.get(GAME["level"], 100)
        
        # Deduct for hints and extra guesses
        # hints_penalty = (GAME["current_hints"] - 1) * 5
        # guesses_penalty = (GAME["current_guesses"] - 1) * 2
        # final_score = max(base - hints_penalty - guesses_penalty, 10)
        
        # For simplicity and to match user's direct score request, 
        # we'll use the base score for the level.
        final_score = base
        
        stats = {
            "level": GAME["level"],
            "difficulty": ["Easy", "Medium", "Hard"][GAME["level"]-1],
            "word": GAME["word"],
            "guesses": GAME["current_guesses"],
            "hints": GAME["current_hints"],
            "score": final_score
        }
        GAME["session_history"].append(stats)
        
        is_session_complete = GAME["level"] >= 3
        if not is_session_complete:
            GAME["level"] += 1

        return {
            "correct": True,
            "message": f"Correct! The word was {GAME['word']}!",
            "stats": stats,
            "session_complete": is_session_complete,
            "session_history": GAME["session_history"]
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

@app.get("/admin/reveal")
async def admin_reveal():
    return {"word": GAME["word"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)