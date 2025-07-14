import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Placeholder for model imports
from yourmodule import (
    extract_video_id,
    get_comments,
    classify_sentiment,
    classify_toxicity_tf,
    detect_emotion
)
# from yourmodule import get_comments, classify_sentiment, classify_toxicity_tf, detect_emotion

# 1. App title and description
st.set_page_config(page_title="YouTube Comment Analyzer", layout="wide")
st.title("💬 YouTube Comment Sentiment & Emotion Analyzer")
st.markdown("Analyze sentiment, toxicity, and emotion of comments from any YouTube video.")

# 2. YouTube video input
video_url = st.text_input("🔗 Paste YouTube Video URL")

# 3. Enter API key (optional or hidden)
api_key = st.text_input("🔑 YouTube API Key", type="password")

# 4. Start button
if st.button("🚀 Analyze Comments") and video_url:
    with st.spinner("Fetching and analyzing comments..."):
        # === PHASE 2: Get Comments ===
        from yourmodule import get_comments
        video_id = extract_video_id(video_url)
        comments = get_comments(video_id, api_key, max_comments=200)
        df = pd.DataFrame({'comment': comments})

        # === PHASE 3: Sentiment ===
        from yourmodule import classify_sentiment
        df['Sentiment'] = df['comment'].apply(classify_sentiment)

        # === PHASE 4: Toxicity (TensorFlow) ===
        from yourmodule import classify_toxicity_tf
        df['Toxicity'] = df['comment'].apply(classify_toxicity_tf)

        # === PHASE 5: Emotion ===
        from yourmodule import detect_emotion
        df['Emotion'] = df['comment'].apply(detect_emotion)

        st.success("✅ Analysis Complete")

        # 5. Show dataframe
        st.dataframe(df.head(20))

        # 6. Visuals
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Sentiment Distribution")
            sent_plot = df['Sentiment'].value_counts()
            st.bar_chart(sent_plot)

        with col2:
            st.markdown("### 😡 Toxic vs Non-Toxic")
            tox_plot = df['Toxicity'].value_counts()
            st.bar_chart(tox_plot)

        st.markdown("### 😶‍🌫️ Emotion Distribution")
        emo_plot = df['Emotion'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=emo_plot.index, y=emo_plot.values, ax=ax, palette='coolwarm')
        ax.set_title("Top Detected Emotions")
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # 7. Top toxic comments
        st.markdown("### ⚠️ Most Toxic Comments")
        st.write(df[df['Toxicity'] == 'Toxic'][['comment']].head(5))

        # 8. Download results
        csv = df.to_csv(index=False)
        st.download_button("📥 Download CSV", csv, file_name="youtube_analysis.csv", mime="text/csv")



