import streamlit as st
import joblib
import spacy
from textblob import TextBlob

@st.cache_resource
def load_artifacts():
    model = joblib.load('sentiment_model.pkl')
    aspect_keywords = joblib.load('aspect_keywords.pkl')
    nlp = spacy.load('en_core_web_sm')
    return model, aspect_keywords, nlp

pipeline, aspect_keywords, nlp = load_artifacts()

def map_chunk_to_aspect(chunk_text):
    chunk_lower = chunk_text.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(kw in chunk_lower for kw in keywords):
            return aspect
    return None

def get_aspect_sentiments(text):
    doc = nlp(text)
    results = []
    for sent in doc.sents:
        for chunk in nlp(sent.text).noun_chunks:
            aspect = map_chunk_to_aspect(chunk.text)
            if aspect:
                polarity = TextBlob(sent.text).sentiment.polarity
                sentiment = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
                results.append({'aspect': aspect, 'phrase': chunk.text, 'sentiment': sentiment})
    return results

st.set_page_config(page_title="Aspect-Based Sentiment Analyzer", page_icon="🔍")
st.title("🔍 Aspect-Based Sentiment Analyzer")
st.write("Analyzes product reviews and identifies sentiment toward specific aspects (battery, screen, price, etc.)")

review_text = st.text_area("Enter a product review:", height=120,
    placeholder="e.g., The screen is great but the battery life is disappointing.")

if st.button("Analyze"):
    if review_text.strip():
        overall = pipeline.predict([review_text])[0]
        st.subheader("Overall Sentiment")
        emoji = {"positive": "😊", "neutral": "😐", "negative": "😞"}
        st.write(f"{emoji.get(overall, '')} **{overall.upper()}**")

        aspects = get_aspect_sentiments(review_text)
        st.subheader("Aspect Breakdown")
        if aspects:
            for a in aspects:
                icon = {"positive": "✅", "neutral": "➖", "negative": "❌"}
                st.write(f"{icon.get(a['sentiment'])} **{a['aspect'].replace('_', ' ').title()}** — {a['sentiment']} (\"{a['phrase']}\")")
        else:
            st.write("No specific product aspects detected in this review.")
    else:
        st.warning("Please enter a review first.")
