import requests
import google.generativeai as genai
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import ExactMatchMetric, BaseMetric


# ===== Gemini setup =====
GEMINI_API_KEY = "AIzaSyB9pbVAZt_q7wmaoIwF**************"
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"


# ===== Call local mock server =====
def call_mock_server(prompt: str) -> str:
    """Send input to local mock FastAPI server"""
    try:
        resp = requests.post("http://127.0.0.1:8000/query", json={"input": prompt}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("output", "").strip()
    except Exception as e:
        print("⚠️ Mock server call failed:", e)
        return ""


# ===== Custom Gemini evaluation metric =====
class GeminiEvalMetric(BaseMetric):
    def __init__(self, name: str, criteria: str):
        self.name = name
        self.criteria = criteria
        self.score = 0.0
        self.reason = ""
        self.threshold = 0.5       # marks pass/fail threshold
        self.async_mode = True
        self.strict_mode = False

    async def a_measure(self, test_case):
        """Async evaluator run by DeepEval"""
        prompt = f"""
You are evaluating a model answer.

Criteria: {self.criteria}

User query: {test_case.input}
Expected answer: {test_case.expected_output}
Model answer: {test_case.actual_output}

Rate the model answer from 0 (poor) to 1 (excellent) based on the criteria.
Then explain briefly.
"""
        try:
            model = genai.GenerativeModel(MODEL)
            result = model.generate_content(prompt)
            text = (result.text or "").strip()

            # Basic keyword-based score extraction
            if "1" in text[:5] or "excellent" in text.lower():
                self.score = 1.0
            elif "0" in text[:5] or "poor" in text.lower():
                self.score = 0.0
            else:
                self.score = 0.5
            self.reason = text
        except Exception as e:
            self.score = 0.0
            self.reason = f"Error during Gemini eval: {e}"

    def is_successful(self) -> bool:
        """Tells DeepEval if metric passes or fails"""
        return self.score >= self.threshold


# ===== Metrics =====
exact_match = ExactMatchMetric(verbose_mode=True)

gemini_relevance = GeminiEvalMetric(
    name="Relevance",
    criteria="Does the answer correctly and completely address the user's query?"
)

gemini_faithfulness = GeminiEvalMetric(
    name="Faithfulness",
    criteria="Is the answer factually consistent with the provided context and free from hallucinations?"
)


# ===== Test cases =====
test_cases = [
    LLMTestCase(
        input="Tell me about crime rate",
        actual_output=call_mock_server("Tell me about crime rate"),
        expected_output="NYPD data shows a 2% drop in overall crime in 2023."
    ),
    LLMTestCase(
        input="Explain about burglary incidents",
        actual_output=call_mock_server("Explain about burglary incidents"),
        expected_output="Burglary incidents were highest in precincts 19 and 23 last year."
    ),
    LLMTestCase(
        input="What community safety initiative launched in 2024?",
        actual_output=call_mock_server("What community safety initiative launched in 2024?"),
        expected_output="Community policing and neighborhood safety programs launched in 2024."
    ),
]


# ===== Run evaluation =====
evaluate(
    test_cases,
    metrics=[exact_match, gemini_relevance, gemini_faithfulness]
)

