# 🧪 DeepEval Test – GEval + Local Mock RAG Server

This project demonstrates how to evaluate LLM responses using **DeepEval** with **Google Gemini** as an evaluator, and a **local FastAPI mock RAG server** to simulate RAG-style responses.

It tests metrics like **Exact Match**, **Relevance**, and **Faithfulness** using both predefined answers and Gemini-based judgment.

---

## 🚀 Features

* Evaluates AI outputs without connecting to a real LLM backend.
* Uses a **mock RAG FastAPI server** returning hardcoded responses for queries.
* Integrates **Gemini** for qualitative evaluation via DeepEval’s custom metric.
* Supports **asynchronous evaluation** and metric-based scoring.

---

## 🧰 Prerequisites

* Python 3.9 or higher
* Google Gemini API key (you can obtain one from [Google AI Studio](https://aistudio.google.com/))

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/deepeval-test-GEval.git
cd deepeval-test-GEval
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

```bash
# On Windows
venv\Scripts\activate

# On macOS / Linux
source venv/bin/activate
```

---

### 3️⃣ Install Required Packages

```bash
pip install deepeval requests google-generativeai fastapi uvicorn
```

---

### 4️⃣ Start the Local Mock RAG Server

This server simulates RAG responses using hardcoded answers.

Run:

```bash
uvicorn local_rag_server:app --port 8000
```

You should see:

```
INFO:     Uvicorn running on http://127.0.0.1:8000
```

You can test the endpoint with:

```bash
Invoke-WebRequest -Uri "http://127.0.0.1:8000/query" `
  -Method POST `
  -Headers @{ "Content-Type" = "application/json" } `
  -Body '{"input": "Tell me about crime rate"}'
```

Expected response:

```json
{"output": "NYPD data shows a 2% drop in overall crime in 2023.", "retrieval_context": ["NYPD data shows a 2% drop in overall crime in 2023."]}
```

---

### 5️⃣ Run the Evaluation Script

```bash
python deepeval_geval_no_rag.py
```

This script:

* Calls the mock server for responses.
* Compares the actual vs. expected answers.
* Uses Gemini-based evaluation for **Relevance** and **Faithfulness**.
* Uses DeepEval’s built-in **Exact Match** metric.

---

## 📊 Example Output

```
Evaluating test cases...
✅ Test 1: Relevance = 1.0, Faithfulness = 1.0, Exact Match = 1.0
✅ Test 2: Relevance = 1.0, Faithfulness = 1.0, Exact Match = 1.0
✅ Test 3: Relevance = 1.0, Faithfulness = 1.0, Exact Match = 1.0
```

---

## 📂 Project Structure

```
deepeval-test-GEval/
│
├── deepeval_geval_no_rag.py        # Main evaluation script
├── local_rag_server.py             # Mock RAG FastAPI server
├── requirements.txt                # Dependencies
└── README.md                       # Project documentation
```

---

## 🧠 Notes

* You can modify the test cases in `deepeval_geval_no_rag.py` to suit your dataset or evaluation goals.
* If you want to connect a real model (like Ollama or OpenAI), replace the mock server API call in `call_mock_server()`.

---
