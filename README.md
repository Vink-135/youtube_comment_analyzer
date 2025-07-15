# 💬 AI Sentiment & Emotion Analyzer for YouTube

> An AI-powered Streamlit app that analyzes **YouTube comments** to extract **sentiment**, **emotion**, and **toxicity** — using state-of-the-art NLP models from HuggingFace and VADER.

---

### 🔍 Demo Features

- 🔗 Paste any **YouTube video URL**
- 📥 Automatically **fetches all comments**
- 🎯 Performs:
  - ✅ **Sentiment Analysis** (Positive, Negative, Neutral)
  - 😡 **Toxic Comment Detection** using **ToxicBERT**
  - ❤️ **Emotion & Subjectivity** with **VADER**
- 📊 Outputs:
  - Interactive **pie charts**
  - Keyword extraction
  - Top toxic or polarizing comments
  - Downloadable CSV

---

### 🤖 Tech Stack

| Component         | Library/Tool                        |
|------------------|-------------------------------------|
| Frontend         | `Streamlit`                         |
| NLP Models       | `HuggingFace Transformers`, `VADER` |
| Toxicity Model   | `ToxicBERT`                         |
| YouTube Comments | `YouTube Data API v3`               |
| Visualization    | `Matplotlib`, `Seaborn`, `Pandas`   |


---

### 📦 Installation (Local)

```bash
git clone https://github.com/yourusername/youtube-sentiment-analyzer.git
cd youtube-sentiment-analyzer
pip install -r requirements.txt
streamlit run ytsentiment.py
