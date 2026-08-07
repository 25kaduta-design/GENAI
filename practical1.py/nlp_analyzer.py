import streamlit as st
import nltk
import spacy
import subprocess
import sys

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------

st.set_page_config(
    page_title="NLP Text Analyzer",
    page_icon="📝",
    layout="wide"
)

# -------------------------------------------------------
# DOWNLOAD NLTK RESOURCES
# -------------------------------------------------------

@st.cache_resource
def download_nltk_resources():
    resources = [
        "punkt",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger"
    ]

    for resource in resources:
        nltk.download(resource, quiet=True)

download_nltk_resources()

# -------------------------------------------------------
# LOAD SPACY MODEL
# -------------------------------------------------------

@st.cache_resource
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_sm"]
        )
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# -------------------------------------------------------
# TITLE
# -------------------------------------------------------

st.title("📝 NLP Text Analyzer")
st.subheader("Name : Tejaswini Amol Kadu")
st.subheader("Roll No : 52")

st.write(
    "Enter a paragraph below and perform different Natural Language Processing operations."
)

# -------------------------------------------------------
# TEXT INPUT
# -------------------------------------------------------

text = st.text_area(
    "Enter your paragraph here:",
    height=200,
    placeholder="Example: Barack Obama was born in Hawaii. He served as the 44th President of the United States."
)

col1, col2 = st.columns(2)

with col1:
    analyze_button = st.button(
        "🔍 Analyze Text",
        use_container_width=True
    )

with col2:
    clear_button = st.button(
        "🗑 Clear",
        use_container_width=True
    )

if clear_button:
    st.rerun()

if analyze_button:

    if text.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")

    else:

        doc = nlp(text)


        st.header("1️⃣ Sentence Segmentation")

        sentences = sent_tokenize(text)

        for i, sentence in enumerate(sentences, start=1):
            st.write(f"**{i}.** {sentence}")

    

        st.header("2️⃣ Word Tokenization")

        words = word_tokenize(text)

        st.write(words)

        st.write(f"**Total Tokens:** {len(words)}")

        # ====================================================
        # 3. STOP WORD REMOVAL
        # ====================================================

        st.header("3️⃣ Stop Word Removal")

        stop_words = set(stopwords.words("english"))

        filtered_words = [
            word for word in words
            if word.lower() not in stop_words
        ]

        st.write(filtered_words)

        # ====================================================
        # 4. STEMMING
        # ====================================================

        st.header("4️⃣ Stemming")

        stemmer = PorterStemmer()

        for word in filtered_words:
            st.write(f"{word} → {stemmer.stem(word)}")

        # ====================================================
        # 5. LEMMATIZATION
        # ====================================================

        st.header("5️⃣ Lemmatization")

        lemmatizer = WordNetLemmatizer()

        for word in filtered_words:
            st.write(f"{word} → {lemmatizer.lemmatize(word)}")

        # ====================================================
        # 6. POS TAGGING
        # ====================================================

        st.header("6️⃣ POS Tagging")

        pos_tags = nltk.pos_tag(words)

        for word, tag in pos_tags:
            st.write(f"**{word}** → `{tag}`")

        # ====================================================
        # 7. NAMED ENTITY RECOGNITION
        # ====================================================

        st.header("7️⃣ Named Entity Recognition")

        if len(doc.ents) == 0:
            st.info("No Named Entities Found.")

        else:
            for entity in doc.ents:
                st.write(f"**{entity.text}** → `{entity.label_}`")

        # ====================================================
        # 8. DEPENDENCY PARSING
        # ====================================================

        st.header("8️⃣ Dependency Parsing")

        st.write("**Word → Dependency → Head**")

        for token in doc:
            st.write(
                f"**{token.text}** → `{token.dep_}` → **{token.head.text}**"
            )

        # ====================================================
        # SUMMARY
        # ====================================================

        st.header("📊 Summary")

        st.write(f"**Number of Sentences:** {len(sentences)}")
        st.write(f"**Number of Tokens:** {len(words)}")
        st.write(f"**Number of Words After Stop-word Removal:** {len(filtered_words)}")
        st.write(f"**Named Entities Found:** {len(doc.ents)}")

        # ====================================================
        # SUCCESS
        # ====================================================

        st.success("✅ NLP Text Analysis Completed Successfully!")
