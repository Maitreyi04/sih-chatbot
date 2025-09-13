# chatbot_rule.py
import json, re
from fastapi import FastAPI, Request
from pydantic import BaseModel
import random
import uvicorn

# instead, initialize empty keywords
KEYWORDS = {}
KEYWORDS_LOWER = {}

with open("crisis_keywords.json", encoding="utf-8") as f:
    CRISIS = [s.lower() for s in json.load(f)]
with open("content_bank.json", encoding="utf-8") as f:
    CONTENT = json.load(f)

# flatten keywords -> lowercase set for fast matching
KEYWORDS_LOWER = {k: set([term.lower() for term in v]) for k,v in KEYWORDS.items()}

app = FastAPI()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+"," ", text)
    text = re.sub(r"[^a-z0-9\s]"," ", text)
    text = re.sub(r"\s+"," ", text).strip()
    return text

def contains_phrase(text, phrase):
    # simple check for phrase presence (phrase may be multi-word)
    return phrase in text

def detect_crisis(text):
    for phrase in CRISIS:
        if contains_phrase(text, phrase):
            return True, phrase
    return False, None

def keyword_match_scores(text):
    scores = {}
    for bucket, kwset in KEYWORDS_LOWER.items():
        score = 0
        for kw in kwset:
            if len(kw) < 3:
                continue
            # count occurrences - simple heuristic
            if contains_phrase(text, kw):
                score += 1
        scores[bucket] = score
    return scores

def pick_bucket(scores):
    # choose bucket with highest score; tie-break randomly among ties
    best_bucket = max(scores, key=lambda k: scores[k])
    if scores[best_bucket] == 0:
        return None
    # check if tie
    best_score = scores[best_bucket]
    ties = [k for k,v in scores.items() if v == best_score]
    return random.choice(ties) if len(ties) > 1 else best_bucket

def choose_response(bucket):
    if not bucket:
        # fallback
        return "I’m not sure I fully understand. Would you like to try a short breathing exercise or talk to a counsellor?"
    if bucket == "Crisis":
        bucket = "Crisis"
    data = CONTENT.get(bucket, {})
    responses = data.get("responses", [])
    return random.choice(responses) if responses else "I’m here to help — would you like to talk to a counsellor?"

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    text = preprocess(req.message)
    # crisis check
    is_crisis, phrase = detect_crisis(text)
    if is_crisis:
        resp = CONTENT.get("Crisis", {}).get("responses", ["I’m concerned about your safety. Please call emergency services."])[0]
        return {"bucket":"Crisis", "response":resp, "matched": phrase}
    # keyword matching
    scores = keyword_match_scores(text)
    bucket = pick_bucket(scores)
    response = choose_response(bucket)
    # return also scores for debugging
    return {"bucket": bucket or "None", "response": response, "scores": scores}

if __name__ == "__main__":
    uvicorn.run("chatbot:app", host="127.0.0.1", port=8000, reload=True)

