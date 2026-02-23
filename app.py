from flask import Flask, render_template, request
import pickle
import os
import requests

app = Flask(__name__)

# ---------- Paths ----------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- Load ML model and vectorizer ----------

with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf_vectorizer = pickle.load(f)

# According to your current behaviour:
# 1 = Plagiarism, 0 = No Plagiarism
PLAGIARISM_LABEL = 1

# ---------- SerpAPI configuration ----------

# ✅ IMPORTANT:
# 1. Go to https://serpapi.com/
# 2. Sign up and get your API key
# 3. Either:
#    - set environment variable SERPAPI_API_KEY
#    - or directly replace "YOUR_SERPAPI_API_KEY" with your key (for college demo only)

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY", "079e81bd3e6b53b83abef7c7448ab73a0d6042c9441c5f2b00ae7c7eaf835f29")


def detect_plagiarism_ml(input_text: str) -> str:
    """
    Uses your trained ML model (SVM + TF-IDF) to detect plagiarism.
    """
    if not input_text.strip():
        return "Please enter some text."

    vectorized_text = tfidf_vectorizer.transform([input_text])
    prediction = model.predict(vectorized_text)[0]
    print("ML raw prediction:", prediction)

    if prediction == PLAGIARISM_LABEL:
        return "Plagiarism Detected"
    else:
        return "No Plagiarism Detected"


def search_web_sources(query: str, num_results: int = 5):
    """
    Uses SerpAPI (Google Search API) to find possible web sources
    that match the given text.
    Returns a list of {title, snippet, link}.
    """
    results = []

    if not query.strip():
        return results

    if not SERPAPI_API_KEY or SERPAPI_API_KEY == "YOUR_SERPAPI_API_KEY":
        # No API key configured – just log and return empty list
        print("⚠ No SerpAPI API key set. Skipping web source detection.")
        return results

    try:
        params = {
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": num_results,
            "hl": "en"
        }

        response = requests.get("https://serpapi.com/search", params=params, timeout=10)
        data = response.json()

        organic_results = data.get("organic_results", [])
        for item in organic_results[:num_results]:
            results.append({
                "title": item.get("title"),
                "snippet": item.get("snippet"),
                "link": item.get("link")
            })

    except Exception as e:
        print("Web search error:", e)

    return results


# ---------- Routes ----------

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    input_text = request.form.get("text", "")

    # 1) ML-based plagiarism result
    ml_result = detect_plagiarism_ml(input_text)

    # 2) Web-source detection via SerpAPI
    web_sources = search_web_sources(input_text)

    return render_template(
        "index.html",
        input_text=input_text,
        result=ml_result,
        web_sources=web_sources
    )


if __name__ == "__main__":
    # debug=True sirf development ke liye
    app.run(debug=True)
