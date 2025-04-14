# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

# Load models and vectorizer
ai_tfidf_vectorizer = joblib.load("ai_tfidf_vectorizer.pkl")
ai_model = joblib.load("ai_model.pkl")
plagiarism_model = joblib.load("xgb_tuned_model.pkl")
doc_tfidf_vectorizer = joblib.load("xgb_tfidf_vectorizer.pkl")

@app.route("/detect-ai", methods=["POST"])
def detect_ai():
    data = request.get_json()
    text = data.get("text", "")
    vectorized = ai_tfidf_vectorizer.transform([text])
    prediction = ai_model.predict(vectorized)[0]
    prob = ai_model.predict_proba(vectorized)[0][1]
    return jsonify({"prediction": int(prediction), "probability": round(prob, 4)})

@app.route("/check-plagiarism", methods=["POST"])
def check_plagiarism():
    try:
        data = request.get_json()
        source_text = data.get("source", "")
        suspicious_text = data.get("suspicious", "")

        print("✔ Received texts")

        source_vec = doc_tfidf_vectorizer.transform([source_text])
        suspicious_vec = doc_tfidf_vectorizer.transform([suspicious_text])

        print("✔ TF-IDF vectorization done")

        cosine_sim = (suspicious_vec @ source_vec.T).toarray()[0][0]
        src_len = len(source_text.split())
        sus_len = len(suspicious_text.split())

        print(f"✔ Features: cosine_sim={cosine_sim}, src_len={src_len}, sus_len={sus_len}")

        from scipy.sparse import hstack
        combined_features = hstack([
            source_vec,
            suspicious_vec,
            [[cosine_sim]],
            [[src_len]],
            [[sus_len]]
        ])

        print(f"✔ Combined feature shape: {combined_features.shape}")

        prediction = plagiarism_model.predict(combined_features)[0]
        prob = plagiarism_model.predict_proba(combined_features)[0][1]

        print("✔ Prediction complete")

        return jsonify({
            "prediction": int(prediction),
            "probability": float(round(prob, 4)),
            # "cosine_similarity": float(round(cosine_sim, 4)),
            # "length_source": int(src_len),
            # "length_suspicious": int(sus_len)
        })


    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({"error": str(e)}), 500

    
# Ensure the app runs on the correct port for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
