import streamlit as st
import joblib
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from textblob import TextBlob

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

@st.cache_resource
def load_artifacts():
    model = joblib.load('sentiment_model.pkl')
    aspect_keywords = joblib.load('aspect_keywords.pkl')
    return model, aspect_keywords

pipeline, aspect_keywords = load_artifacts()

def map_chunk_to_aspect(phrase):
    phrase_lower = phrase.lower()
    for aspect, keywords in aspect_keywords.items():
        if any(kw in phrase_lower for kw in keywords):
            return aspect
    return None

def extract_noun_phrases(sentence):
    """Simple noun phrase extraction: consecutive nouns, optionally preceded by a determiner/adjective."""
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    phrases = []
    current = []
    for word, tag in tagged:
        if tag.startswith('NN') or tag in ('DT', 'JJ'):
            current.append(word)
        else:
            if current and any(t.startswith('NN') for t, _ in pos_tag(current)):
                phrases.append(' '.join(current))
            current = []
    if current and any(t.startswith('NN') for t, _ in pos_tag(current)):
        phrases.append(' '.join(current))
    return phrases

def get_aspect_sentiments(text):
    results = []
    for sent in sent_tokenize(text):
        phrases = extract_noun_phrases(sent)
        for phrase in phrases:
            aspect = map_chunk_to_aspect(phrase)
            if aspect:
                polarity = TextBlob(sent).sentiment.polarity
                sentiment = 'positive' if polarity > 0.1 else 'negative' if polarity < -0.1 else 'neutral'
                results.append({'aspect': aspect, 'phrase': phrase, 'sentiment': sentiment})
    return results

# --- UI ---
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
