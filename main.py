import streamlit as st
from transformers import pipeline
from textblob import TextBlob
from deep_translator import GoogleTranslator
from gtts import gTTS
import os
import tempfile
import fitz  # PyMuPDF for PDF text extraction
import requests

st.set_page_config(page_title="OmniAI Hub", page_icon="💎", layout="wide")

# ---------------------------------------
# 🌐 Load Models (CPU-Friendly)
# ---------------------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device_map="auto")
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws", device_map="auto")
    chatbot = pipeline("text2text-generation", model="google/flan-t5-base", device_map="auto")
    grammar_model = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1", device_map="auto")
    return summarizer, paraphraser, chatbot, grammar_model

summarizer, paraphraser, chatbot, grammar_model = load_models()

# ---------------------------------------
# ✍️ Grammar Correction Functions
# ---------------------------------------
def correct_grammar_simple(text):
    blob = TextBlob(text)
    return str(blob.correct())

def correct_grammar_advanced(text):
    try:
        result = grammar_model(text, max_length=128, do_sample=False)
        return result[0]['generated_text']
    except Exception as e:
        return f"⚠️ Error in grammar correction: {e}"

def correct_grammar(text, mode="simple"):
    if mode == "simple":
        return correct_grammar_simple(text)
    else:
        return correct_grammar_advanced(text)

# ---------------------------------------
# 🧠 Helper Functions
# ---------------------------------------
def summarize_text(text):
    return summarizer(text, max_length=150, min_length=40, do_sample=False)[0]['summary_text']

def paraphrase_text(text):
    return paraphraser(text, max_length=100, do_sample=False)[0]['generated_text']

def chat_response(prompt):
    return chatbot(prompt, max_length=128, do_sample=False)[0]['generated_text']

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def extract_text_from_pdf(uploaded_file):
    pdf_text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            pdf_text += page.get_text()
    return pdf_text

def text_to_speech(text, lang="en"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
        tts = gTTS(text, lang=lang)
        tts.save(tmpfile.name)
        return tmpfile.name

# ---------------------------------------
# 🏠 Sidebar
# ---------------------------------------
st.sidebar.title("💎 OmniAI Hub")
tool = st.sidebar.radio("Choose Tool", [
    "📄 Summarizer",
    "🌀 Paraphraser",
    "💬 Chatbot",
    "🗣 Translator + Voice",
    "🧾 Grammar Corrector",
    "📚 PDF Summarizer"
])

# ---------------------------------------
# 🧩 Tool Sections
# ---------------------------------------

if tool == "📄 Summarizer":
    st.header("📄 AI Summarizer")
    text = st.text_area("Enter text to summarize:")
    if st.button("Summarize"):
        if text.strip():
            summary = summarize_text(text)
            st.success(summary)
        else:
            st.warning("Please enter text to summarize.")

elif tool == "🌀 Paraphraser":
    st.header("🌀 AI Paraphraser")
    text = st.text_area("Enter text to paraphrase:")
    if st.button("Paraphrase"):
        if text.strip():
            para = paraphrase_text(text)
            st.success(para)
        else:
            st.warning("Please enter some text.")

elif tool == "💬 Chatbot":
    st.header("💬 AI Chatbot")
    prompt = st.text_area("Ask me anything:")
    if st.button("Ask"):
        if prompt.strip():
            answer = chat_response(prompt)
            st.info(answer)
        else:
            st.warning("Please ask a question.")

elif tool == "🗣 Translator + Voice":
    st.header("🌐 Translator & Voice Generator")
    text = st.text_area("Enter text:")
    target_lang = st.text_input("Target language (e.g., urdu, french, german, hindi):")
    if st.button("Translate & Speak"):
        if text.strip() and target_lang.strip():
            translated = translate_text(text, target_lang)
            st.success(f"**Translated:** {translated}")
            audio_path = text_to_speech(translated, lang="en")
            st.audio(audio_path)
        else:
            st.warning("Please enter text and target language.")

elif tool == "🧾 Grammar Corrector":
    st.header("🧾 Grammar Correction Tool")
    text = st.text_area("Enter your text to correct grammar:")
    mode = st.radio("Choose Mode:", ["simple", "advanced"])
    if st.button("Correct Grammar"):
        if text.strip():
            corrected = correct_grammar(text, mode)
            st.success(f"**Corrected Text:**\n\n{corrected}")
        else:
            st.warning("Please enter text first.")

elif tool == "📚 PDF Summarizer":
    st.header("📚 PDF Summary Extractor")
    uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text:", pdf_text[:2000])
        if st.button("Summarize PDF"):
            summary = summarize_text(pdf_text)
            st.success(summary)
