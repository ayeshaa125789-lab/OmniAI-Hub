import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator
from gtts import gTTS
import fitz  # PyMuPDF
import os
import tempfile

# --------------------------- #
# 🎯 APP CONFIG
# --------------------------- #
st.set_page_config(page_title="OmniAI Hub", page_icon="💎", layout="wide")

st.title("💎 OmniAI Hub — All-in-One Free AI Assistant")
st.caption("Summarize • Paraphrase • Translate • Grammar Check • Chat • PDF Extract • Voice")

# --------------------------- #
# 🧩 LOAD MODELS (lightweight)
# --------------------------- #
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    chatbot = pipeline("text-generation", model="microsoft/DialoGPT-small")
    grammar_fixer = pipeline("text2text-generation", model="prithivida/grammar_error_correcter_v1")
    return summarizer, paraphraser, chatbot, grammar_fixer

summarizer, paraphraser, chatbot, grammar_fixer = load_models()

# --------------------------- #
# ⚙️ HELPER FUNCTIONS
# --------------------------- #
def summarize_text(text):
    summary = summarizer(text, max_length=150, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def paraphrase_text(text):
    para = paraphraser(f"paraphrase: {text}", max_length=200, do_sample=False)
    return para[0]['generated_text']

def chat_with_ai(prompt):
    response = chatbot(prompt, max_length=100, do_sample=True)
    return response[0]['generated_text']

def grammar_check_text(text):
    fixed = grammar_fixer(f"grammar: {text}", max_length=200, do_sample=False)
    return fixed[0]['generated_text']

def translate_text(text, target_lang):
    return GoogleTranslator(source='auto', target=target_lang).translate(text)

def extract_pdf_text(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def text_to_speech(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        return tmp.name

# --------------------------- #
# 🎛️ SIDEBAR MENU
# --------------------------- #
menu = st.sidebar.selectbox(
    "Choose a Tool",
    [
        "📄 Summarizer",
        "✍️ Paraphraser",
        "🔠 Grammar Checker",
        "💬 Chatbot",
        "🌍 Translator",
        "📘 PDF/Text Extractor",
        "🎙️ Text to Voice"
    ]
)

# --------------------------- #
# 🧩 TOOLS SECTION
# --------------------------- #
if menu == "📄 Summarizer":
    st.subheader("Summarize Any Text or PDF")
    uploaded = st.file_uploader("Upload PDF (optional)", type=["pdf"])
    text = st.text_area("Enter text to summarize")

    if uploaded:
        text = extract_pdf_text(uploaded)
        st.info("✅ PDF text extracted successfully!")

    if st.button("Summarize"):
        if text.strip():
            st.success(summarize_text(text))
        else:
            st.warning("Please provide text or upload a PDF.")

elif menu == "✍️ Paraphraser":
    st.subheader("AI Text Rewriter (Free Quillbot Alternative)")
    text = st.text_area("Enter text to paraphrase")
    if st.button("Paraphrase"):
        if text.strip():
            st.success(paraphrase_text(text))
        else:
            st.warning("Enter some text first!")

elif menu == "🔠 Grammar Checker":
    st.subheader("AI Grammar & Spell Checker")
    text = st.text_area("Enter text to fix grammar")
    if st.button("Fix Grammar"):
        if text.strip():
            st.success(grammar_check_text(text))
        else:
            st.warning("Enter text first.")

elif menu == "💬 Chatbot":
    st.subheader("Chat with SmartBot 🤖")
    user_input = st.text_input("Ask anything...")
    if st.button("Chat"):
        if user_input.strip():
            st.info(chat_with_ai(user_input))
        else:
            st.warning("Please type something!")

elif menu == "🌍 Translator":
    st.subheader("AI Multi-Language Translator 🌎")
    text = st.text_area("Enter text to translate")
    lang = st.selectbox("Choose target language", ["en", "ur", "hi", "fr", "es", "ar", "zh-cn"])
    if st.button("Translate"):
        if text.strip():
            st.success(translate_text(text, lang))
        else:
            st.warning("Please enter text first.")

elif menu == "📘 PDF/Text Extractor":
    st.subheader("Extract Text from PDF or Upload Document")
    file = st.file_uploader("Upload PDF file", type=["pdf"])
    if file:
        text = extract_pdf_text(file)
        st.text_area("Extracted Text", text, height=250)

elif menu == "🎙️ Text to Voice":
    st.subheader("Convert Text to Speech 🎧")
    text = st.text_area("Enter text to convert")
    if st.button("Generate Voice"):
        if text.strip():
            path = text_to_speech(text)
            st.audio(path)
        else:
            st.warning("Enter text first.")

