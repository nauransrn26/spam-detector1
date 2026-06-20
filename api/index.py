from flask import Flask, render_template, request
import joblib

from docx import Document
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(
    __name__,
    template_folder="../templates",
    static_folder="../static"
)

# =========================
# Load Model
# =========================

import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

model = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "spam_model.pkl"
    )
)

tfidf = joblib.load(
    os.path.join(
        BASE_DIR,
        "model",
        "tfidf.pkl"
    )
)

print("Classes:", model.classes_)

# =========================
# Sastrawi
# =========================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = stemmer.stem(text)
    return text

# =========================
# Read DOCX
# =========================

def read_docx(file):
    doc = Document(file)

    text = ""

    for para in doc.paragraphs:
        text += para.text + " "

    return text

# =========================
# Landing Page
# =========================

@app.route("/")
def landing():
    return render_template("landing.html")

# =========================
# Detector Page
# =========================

@app.route("/detector")
def detector():
    return render_template("index.html")

# =========================
# Prediction
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    text = request.form.get("text", "")

    uploaded_file = request.files.get("file")

    if uploaded_file and uploaded_file.filename != "":
        text = read_docx(uploaded_file)

    processed_text = preprocess(text)

    vector = tfidf.transform([processed_text])

    feature_names = tfidf.get_feature_names_out()

    tfidf_values = vector.toarray()[0]

    tfidf_result = []

    for i, value in enumerate(tfidf_values):
        if value > 0:
            tfidf_result.append({
                "term": feature_names[i],
                "weight": round(float(value), 4)
            })

    tfidf_result = sorted(
        tfidf_result,
        key=lambda x: x["weight"],
        reverse=True
    )[:10]

    prediction = model.predict(vector)[0]

    probabilities = model.predict_proba(vector)[0]

    classes = model.classes_

    prob_dict = {}

    for i, label in enumerate(classes):
        prob_dict[label] = round(
            probabilities[i] * 100,
            2
        )

    spam_prob = prob_dict.get("spam", 0)
    nonspam_prob = prob_dict.get("non-spam", 0)

    return render_template(
        "index.html",
        prediction=prediction,
        spam_prob=spam_prob,
        nonspam_prob=nonspam_prob,
        text=text,
        tfidf_result=tfidf_result
    )

# =========================
# Vercel Entry Point
# =========================

def handler(environ, start_response):
    return app(environ, start_response)