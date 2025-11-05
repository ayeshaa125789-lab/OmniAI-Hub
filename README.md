# 💎 OmniAI Hub

OmniAI Hub is a **free, all-in-one AI assistant** built with [Streamlit](https://streamlit.io).  
It combines multiple smart tools — summarizer, grammar checker, paraphraser, plagiarism detector, chatbot, translator, and voice features — all powered by **free open-source AI models**.  
Perfect for students, teachers, writers, and professionals.

---

## 🚀 Features

| Category | Description |
|-----------|--------------|
| 📄 **PDF Summarizer** | Extracts and summarizes long PDF files using T5-small |
| ✍️ **Grammar & Paraphraser** | Checks grammar and rewrites text (like QuillBot & Grammarly) |
| 🔎 **Plagiarism Checker** | Compares text similarity using TF-IDF & cosine similarity |
| 💬 **AI Chatbot** | Gives instant responses using FLAN-T5 |
| 🌍 **Translator & Voice** | Translate text into English, Urdu, or Hindi + Speak or Voice-to-Text |
| 💾 **Text Downloader** | Download results as TXT, DOCX, or PDF |
| 🔉 **Speech Recognition** | Works with Google Speech Recognition (auto-uses Whisper if available) |

---

## 🧠 Tech Stack

- **Streamlit** – Interactive web interface  
- **Transformers (Hugging Face)** – Summarization, paraphrasing, chatbot  
- **LanguageTool** – Grammar correction  
- **Scikit-learn** – Text similarity & plagiarism check  
- **PyMuPDF** – PDF text extraction  
- **gTTS / SpeechRecognition / Whisper (optional)** – Text-to-Speech & Voice Input  
- **Deep Translator** – Multilingual translation  
- **ReportLab & python-docx** – File downloads  

---

## 🛠️ Installation & Usage

### 1️⃣ Clone or Download the Project
```bash
git clone https://github.com/<your-username>/OmniAI-Hub.git
cd OmniAI-Hub
