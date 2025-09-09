import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import re

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Sentiment Analysis Dashboard")
st.markdown("""
Analyze sentiment of your dataset.
- Upload CSV with text columns
- See **positive, neutral, negative** sentiment
- Visualize with **Bar chart, Pie chart, and Count graph**
- Filter and download filtered data
""")

# ----------------------------
# File upload
# ----------------------------
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # ----------------------------
    # Detect text columns and merge
    # ----------------------------
    text_cols = df.select_dtypes(include=["object"]).columns  # all string columns
    if len(text_cols) == 0:
        st.error("No text columns found in your CSV.")
        st.stop()

    # Merge all text columns into one string per row
    df["Combined_Text"] = df[text_cols].astype(str).agg(" ".join, axis=1)

    # ----------------------------
    # Clean Text
    # ----------------------------
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)  # remove links
        text = re.sub(r"@\w+", "", text)            # remove mentions
        text = re.sub(r"#\w+", "", text)            # remove hashtags
        text = re.sub(r"[^a-z\s]", "", text)        # remove punctuation/numbers
        text = re.sub(r"\s+", " ", text).strip()    # remove extra spaces
        return text

    df["Cleaned_Text"] = df["Combined_Text"].apply(clean_text)

    # ----------------------------
    # Sentiment Analysis
    # ----------------------------
    def get_sentiment(text):
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    df["Sentiment"] = df["Cleaned_Text"].apply(get_sentiment)

    # ----------------------------
    # Metrics
    # ----------------------------
    total = len(df)
    pos = len(df[df["Sentiment"]=="Positive"])
    neg = len(df[df["Sentiment"]=="Negative"])
    neu = len(df[df["Sentiment"]=="Neutral"])

    # ----------------------------
    # Dashboard Tabs
    # ----------------------------
    tab1, tab2, tab3 = st.tabs(["Summary", "Charts", "Filtered Data"])

    # Tab 1: Summary Metrics
    with tab1:
        st.subheader("📊 Sentiment Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Rows", total)
        col2.metric("Positive", pos, delta=f"{round(pos/total*100,1)}%")
        col3.metric("Neutral", neu, delta=f"{round(neu/total*100,1)}%")
        col4.metric("Negative", neg, delta=f"{round(neg/total*100,1)}%")

    # Tab 2: Charts
    with tab2:
        st.subheader("📊 Sentiment Charts")
        col1, col2 = st.columns(2)
        
        # Bar chart
        with col1:
            fig_bar, ax_bar = plt.subplots(figsize=(4,3))
            df["Sentiment"].value_counts().plot(kind="bar", color=["green","grey","red"], ax=ax_bar)
            ax_bar.set_ylabel("Count")
            ax_bar.set_xlabel("Sentiment")
            ax_bar.set_title("Sentiment Distribution")
            st.pyplot(fig_bar)
        
        # Pie chart
        with col2:
            fig_pie, ax_pie = plt.subplots(figsize=(4,3))
            df["Sentiment"].value_counts().plot(
                kind="pie", autopct="%1.1f%%", colors=["green","grey","red"], startangle=90, ax=ax_pie
            )
            ax_pie.set_ylabel("")
            st.pyplot(fig_pie)

        # Count graph (line plot)
        st.subheader("📈 Count Graph")
        fig_line, ax_line = plt.subplots(figsize=(8,3))
        sentiment_counts = df["Sentiment"].value_counts()
        ax_line.plot(sentiment_counts.index, sentiment_counts.values, marker='o', linestyle='-', color='blue')
        ax_line.set_xlabel("Sentiment")
        ax_line.set_ylabel("Count")
        ax_line.set_title("Sentiment Count")
        st.pyplot(fig_line)

    # Tab 3: Filter & Download
    with tab3:
        st.subheader("🔍 Filter Rows")
        sentiment_choice = st.selectbox("Select Sentiment", ["All","Positive","Neutral","Negative"])
        
        if sentiment_choice != "All":
            filtered_df = df[df["Sentiment"]==sentiment_choice]
        else:
            filtered_df = df

        st.dataframe(filtered_df, height=300)

        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=filtered_df.to_csv(index=False).encode('utf-8'),
            file_name='filtered_reviews.csv',
            mime='text/csv'
        )
