from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class QueryRequest(BaseModel):
    input: str

@app.post("/query")
def query_rag(req: QueryRequest):
    query = req.input.lower()

    # ---- Mock retrieval store ----
    retrieval_db = {
        "crime rate": ["NYPD data shows a 2% drop in overall crime in 2023."],
        "burglary": ["Burglary incidents were highest in precincts 19 and 23 last year."],
        "initiative": ["Community policing and neighborhood safety programs launched in 2024."],
        "larceny": ["Grand larceny in New York is theft over $1,000 (Penal Law Article 155)."],
    }

    # ---- Mock answer store (simulated LLM output) ----
    answer_db = {
        "crime rate": "The crime rate in New York dropped by about 2% in 2023, mainly due to reduced burglaries and assaults.",
        "burglary": "Burglary incidents peaked in precincts 19 and 23, but overall numbers declined compared to 2022.",
        "initiative": "Several community safety programs were launched in 2024 to improve neighborhood policing.",
        "larceny": "Grand larceny in New York refers to theft over $1,000, as defined under Penal Law Article 155.",
    }

    # ---- Retrieve context and hardcoded answer ----
    context = []
    answer = "I don't have data on that topic. Please refine your question."

    for key, docs in retrieval_db.items():
        if key in query:
            context = docs
            answer = answer_db[key]
            break

    return {
        "output": answer,
        "retrieval_context": context
    }
