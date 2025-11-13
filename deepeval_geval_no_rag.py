import requests
import google.generativeai as genai
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric, BaseMetric


# ===== GEMINI CONFIGURATION =====
GEMINI_API_KEY = "AIzaSyB9pbVAZt_q7wmao**************************************"
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"


# ===== LOCAL MOCK SERVER CALL =====
def call_mock_server(prompt: str) -> dict:
    """Send input to local mock FastAPI server and return output + context."""
    try:
        resp = requests.post("http://127.0.0.1:8000/query", json={"input": prompt}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "output": data.get("output", "").strip(),
            "context": data.get("retrieval_context", [])
        }
    except Exception as e:
        print("⚠️ Mock server call failed:", e)
        return {"output": "", "context": []}


# ===== CUSTOM GEMINI EVAL METRIC (GENERIC CLASS) =====
class GeminiEvalMetric(BaseMetric):
    def __init__(self, name: str, criteria: str, threshold: float = 0.5):
        self.name = name
        self.criteria = criteria
        self.score = 0.0
        self.reason = ""
        self.threshold = threshold
        self.async_mode = True
        self.strict_mode = False

    async def a_measure(self, test_case):
        """Asynchronous metric evaluation using Gemini."""
        # Combine all retrieval context if available
        context_data = getattr(test_case, "retrieval_context", [])
        if isinstance(context_data, list):
            context_data = " | ".join(context_data)

        prompt = f"""
You are evaluating a model answer based on the following criteria.

**Criteria:** {self.criteria}

User query: {test_case.input}
Retrieved context: {context_data}
Expected answer: {test_case.expected_output}
Model answer: {test_case.actual_output}

Rate the answer from 0 (poor) to 1 (excellent) and provide a short reason.
"""
        try:
            model = genai.GenerativeModel(MODEL)
            result = model.generate_content(prompt)
            text = (result.text or "").strip()

            # crude numeric extraction heuristic
            if "1" in text[:5] or "excellent" in text.lower():
                self.score = 1.0
            elif "0" in text[:5] or "poor" in text.lower():
                self.score = 0.0
            elif "0.8" in text or "0.9" in text:
                self.score = 0.9
            elif "0.6" in text or "0.7" in text:
                self.score = 0.7
            else:
                self.score = 0.5

            self.reason = text
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error during Gemini eval: {e}"

    def is_successful(self) -> bool:
        return self.score >= self.threshold


# ===== METRICS DEFINITIONS =====
exact_match = ExactMatchMetric(verbose_mode=True)

gemini_relevance = GeminiEvalMetric(
    name="Relevance",
    criteria="Does the answer directly and completely address the user's question?"
)

gemini_faithfulness = GeminiEvalMetric(
    name="Faithfulness",
    criteria="Is the answer factually consistent with the retrieved context and free of hallucinations?"
)

# --- RAGAS-style Gemini metrics ---
gemini_context_relevance = GeminiEvalMetric(
    name="Context Relevance",
    criteria="Does the retrieved context contain information that is directly useful to answer the query?"
)

gemini_context_precision = GeminiEvalMetric(
    name="Context Precision",
    criteria="Among all retrieved context, how much is actually relevant and non-redundant for the query?"
)

gemini_context_recall = GeminiEvalMetric(
    name="Context Recall",
    criteria="Does the context include all necessary information needed to fully answer the user's question?"
)


# ===== TEST CASES =====
cases_data = [
    ("Tell me about crime rate", "NYPD data shows a 2% drop in overall crime in 2023."),
    ("Explain about burglary incidents", "Burglary incidents were highest in precincts 19 and 23 last year."),
    ("What community safety initiative launched in 2024?", "Community policing and neighborhood safety programs launched in 2024.")
]

test_cases = []
for user_input, expected in cases_data:
    result = call_mock_server(user_input)
    case = LLMTestCase(
        input=user_input,
        actual_output=result["output"],
        expected_output=expected
    )
    # Attach context manually (safe for all deepeval versions)
    case.retrieval_context = result["context"]
    test_cases.append(case)


# ===== RUN EVALUATION =====
evaluate(
    test_cases,
    metrics=[
        exact_match,
        gemini_relevance,
        gemini_faithfulness,
        gemini_context_relevance,
        gemini_context_precision,
        gemini_context_recall
    ]
)
