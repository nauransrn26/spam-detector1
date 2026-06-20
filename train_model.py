import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =========================
# Sastrawi Stemmer
# =========================

factory = StemmerFactory()
stemmer = factory.create_stemmer()

def preprocess(text):
    text = str(text).lower()
    text = stemmer.stem(text)
    return text

# =========================
# Load Dataset
# =========================

df = pd.read_csv("dataset/spam.csv")

print("Kolom Dataset:")
print(df.columns)

print("\nJumlah Data:")
print(df.shape)

# =========================
# Preprocessing
# =========================

df["text"] = df["text"].apply(preprocess)

# =========================
# Feature dan Label
# =========================

X = df["text"]
y = df["label"]

# =========================
# TF-IDF
# =========================

tfidf = TfidfVectorizer()

X_tfidf = tfidf.fit_transform(X)

# =========================
# Training Model
# =========================

model = MultinomialNB()

model.fit(X_tfidf, y)

# =========================
# Save Model
# =========================

joblib.dump(
    model,
    "model/spam_model.pkl"
)

joblib.dump(
    tfidf,
    "model/tfidf.pkl"
)

print("\nModel berhasil disimpan!")