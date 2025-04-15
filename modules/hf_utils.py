from transformers import pipeline

# ⚠️ Generic sentiment model used to simulate fake/real
classifier = pipeline("sentiment-analysis")

def hf_classify(tweet):
    result = classifier(tweet)[0]
    label = "FAKE" if result["label"] == "NEGATIVE" else "REAL"
    return label, round(result["score"] * 100, 1)
