import streamlit as st
import fitz  # PyMuPDF
import os
import tempfile
import language_tool_python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from deep_translator import GoogleTranslator
from gtts import gTTS
from io import BytesIO
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import speech_recognition as sr

# ------------------------------
# Page setup
# ------------------------------
st.set_page_config(page_title="Omni AI Hub", page_icon="🤖", layout="wide")
st.title("💎 Omni AI Hub – All-in-One Smart Assistant (Free & Open Source)")

# ------------------------------
# Initialize models
# ------------------------------
@st.cache_resource
def load_models():
    summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
    paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")
    chatbot = pipeline("text2text-generation", model="google/flan-t5-small")
    return summarizer, paraphraser, chatbot

summarizer, paraphraser, chatbot = load_models()
tool = language_tool_python.LanguageTool('en-US')

# ------------------------------
# Helper Functions
# ------------------------------
def extract_text_from_pdf(file):
    text = ""
    with fitz.open(stream=file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def summarize_text(text):
    if len(text.split()) < 30:
        return "Text too short to summarize."
    chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
    result = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        result.append(summary)
    return " ".join(result)

def paraphrase_text(text):
    output = paraphraser(f"paraphrase: {text}", max_length=200, num_return_sequences=1)
    return output[0]['generated_text']

def grammar_check(text):
    matches = tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected

def check_plagiarism(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    similarity = cosine_similarity(vectorizer)[0][1]
    return round(similarity * 100, 2)

def chatbot_reply(prompt):
    response = chatbot(prompt, max_length=100)[0]['generated_text']
    return response

def translate_text(text, target_lang):
    translated = GoogleTranslator(source='auto', target=target_lang).translate(text)
    return translated

def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    buf = BytesIO()
    tts.write_to_fp(buf)
    return buf

def transcribe_voice():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Speak now...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except:
        return "Sorry, I couldn't understand the voice."

# ------------------------------
# Sidebar Navigation
# ------------------------------
menu = st.sidebar.selectbox("🧩 Choose a Feature", [
    "📄 PDF Summarizer",
    "✍️ Grammar & Paraphraser",
    "🔎 Plagiarism Checker",
    "💬 AI Chatbot",
    "🌍 Translator & Voice",
    "🧾 Text Downloader"
])

# ------------------------------
# PDF Summarizer
# ------------------------------
if menu == "📄 PDF Summarizer":
    st.subheader("📚 Upload a PDF and get summary")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file:
        text = extract_text_from_pdf(pdf_file)
        st.text_area("Extracted Text", text[:1000] + "..." if len(text) > 1000 else text, height=200)
        if st.button("Summarize"):
            summary = summarize_text(text)
            st.success(summary)

# ------------------------------
# Grammar & Paraphraser
# ------------------------------
elif menu == "✍️ Grammar & Paraphraser":
    st.subheader("🧠 Grammar Correction & Paraphrasing")
    text = st.text_area("Enter text:")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Grammar Check"):
            corrected = grammar_check(text)
            st.success(corrected)
    with col2:
        if st.button("Paraphrase"):
            result = paraphrase_text(text)
            st.info(result)

# ------------------------------
# Plagiarism Checker
# ------------------------------
elif menu == "🔎 Plagiarism Checker":
    st.subheader("🔍 Compare Two Texts")
    text1 = st.text_area("Text 1")
    text2 = st.text_area("Text 2")
    if st.button("Check Similarity"):
        sim = check_plagiarism(text1, text2)
        st.write(f"Similarity: **{sim}%**")

# ------------------------------
# AI Chatbot
# ------------------------------
elif menu == "💬 AI Chatbot":
    st.subheader("🤖 Ask me anything")
    user_input = st.text_input("Your question:")
    if st.button("Get Answer"):
        st.info(chatbot_reply(user_input))

# ------------------------------
# Translator & Voice
# ------------------------------
elif menu == "🌍 Translator & Voice":
    st.subheader("🗣️ Translate, Speak, or Record Voice")
    lang_choice = st.selectbox("Select Language", {"English": "en", "Urdu": "ur", "Hindi": "hi"})
    input_text = st.text_area("Enter text:")
    if st.button("Translate"):
        trans = translate_text(input_text, lang_choice)
        st.success(trans)

    if st.button("Speak"):
        buf = text_to_speech(input_text, lang=lang_choice)
        st.audio(buf.getvalue(), format='audio/mp3')

    if st.button("🎤 Voice to Text"):
        text = transcribe_voice()
        st.write(f"🗣 You said: {text}")

# ------------------------------
# Text Downloader
# ------------------------------
elif menu == "🧾 Text Downloader":
    st.subheader("📦 Download Text as File")
    text = st.text_area("Enter text:")
    filetype = st.selectbox("Choose format", ["TXT", "DOCX", "PDF"])
    if st.button("Download"):
        if filetype == "TXT":
            st.download_button("Download TXT", text, file_name="output.txt")
        elif filetype == "DOCX":
            doc = Document()
            doc.add_paragraph(text)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc.save(tmp.name)
                st.download_button("Download DOCX", tmp.read(), file_name="output.docx")
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                c = canvas.Canvas(tmp.name, pagesize=letter)
                c.drawString(100, 700, text[:1000])
                c.save()
                st.download_button("Download PDF", tmp.read(), file_name="output.pdf")

st.sidebar.markdown("---")
st.sidebar.info("Made with ❤️ using Streamlit + HuggingFace + LanguageTool")
