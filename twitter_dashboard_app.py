# -------------------------------------------------
# Twitter Sentiment Analysis Dashboard (Dark Theme)
# Author: Dharsh
# -------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

# ----------------------------------------
# Page Setup
# ----------------------------------------
st.set_page_config(
    page_title="Twitter Sentiment Dashboard",
    layout="wide",
    page_icon="💬",
)

# ----------------------------------------
# Custom Dark Theme CSS
# ----------------------------------------
st.markdown("""
<style>
    /* App Base */
    body, .stApp {
        background-color: #0e1117;
        color: #ffffff;
        font-family: 'Poppins', sans-serif;
        font-size: 18px;
    }

    /* Main Title */
    h1 {
        color: #00BFFF !important;
        text-shadow: 0 0 25px #00BFFF, 0 0 10px #0099ff;
        font-size: 3em !important;
        text-align: center;
        font-weight: 800;
    }

    h2, h3, h4 {
        color: #00BFFF !important;
        text-shadow: 0 0 10px #00BFFF;
    }

    /* Backgrounds & Containers */
    .block-container {
        background-color: #0e1117;
        color: white;
    }
    div[data-testid="stVerticalBlock"] {
        background-color: transparent !important;
    }

    /* DataFrame & Table */
    .stDataFrame, .stTable {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 10px;
        border: 1px solid #00BFFF;
        font-size: 17px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161a23 !important;
        border-right: 2px solid #00BFFF;
        box-shadow: 0 0 20px #00BFFF;
    }

    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] span {
        color: #e6f0ff !important;
        font-size: 18px !important;
        font-weight: 500 !important;
        text-shadow: 0 0 4px #00BFFF;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        background-color: transparent !important;
        border: 2px solid #00BFFF !important;
        border-radius: 12px !important;
        padding: 14px !important;
        text-align: center !important;
        box-shadow: 0 0 10px #00BFFF;
    }

    /* Drag & Drop area */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 12px rgba(0, 191, 255, 0.4);
        transition: all 0.3s ease-in-out;
    }

    [data-testid="stFileUploaderDropzone"] div,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] small {
        color: #000000 !important;
        font-weight: 600 !important;
        text-shadow: none !important;
    }

    [data-testid="stFileUploaderDropzone"]:hover {
        background-color: #f5f5f5 !important;
        transform: scale(1.02);
        box-shadow: 0 0 15px #00BFFF;
    }

    /* Browse Files Button */
    [data-testid="stFileUploaderDropzone"] div div div div {
        background-color: #00BFFF !important;
        color: #ffffff !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 14px !important;
        text-align: center !important;
        transition: all 0.3s ease-in-out;
        box-shadow: 0 0 10px #00BFFF !important;
    }

    [data-testid="stFileUploaderDropzone"] div div div div:hover {
        background-color: #0099cc !important;
        transform: scale(1.05);
        box-shadow: 0 0 18px #00BFFF !important;
    }

    /* Data Tables */
    .stDataFrame, .stTable, table, th, td {
        color: #ffffff !important;
        font-weight: 600 !important;
        border-color: #00BFFF !important;
        background-color: #12151c !important;
        text-align: center !important;
    }

    /* Graph Labels */
    .plotly text, g.cartesianlayer text, g.xtick text, g.ytick text {
        fill: #000000 !important;
        font-weight: 600 !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00BFFF !important;
        color: #ffffff !important;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-size: 18px;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
        box-shadow: 0 0 10px #00BFFF;
    }

    .stButton>button:hover {
        background-color: #0099cc !important;
        transform: scale(1.05);
        box-shadow: 0 0 18px #00BFFF;
    }

    /* Textareas & Inputs */
    textarea, .stTextInput>div>div>input {
        background-color: #12151c !important;
        color: #ffffff !important;
        border: 1px solid #00BFFF !important;
        border-radius: 8px !important;
        font-size: 17px !important;
        padding: 12px !important;
    }

    /* Dropdown Fix */
    div[data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #00BFFF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        color: #000000 !important;
        background-color: #ffffff !important;
    }

    ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid #00BFFF !important;
        border-radius: 6px !important;
    }

    ul[role="listbox"] li {
        color: #000000 !important;
        background-color: #ffffff !important;
        font-weight: 600 !important;
    }

    ul[role="listbox"] li:hover {
        background-color: #e0f7ff !important;
        color: #000000 !important;
    }

    ul[role="listbox"] li[aria-selected="true"] {
        background-color: #00BFFF !important;
        color: #ffffff !important;
    }

    /* =============================
       ⚙️ Fix: Column Data Text Visibility (Black)
    ============================= */
    div[data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] div,
    div[data-testid="stDataFrame"] table,
    div[data-testid="stDataFrame"] td,
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] span,
    div[data-testid="stDataFrame"] p {
        color: #FFFFFF !important;
        background-color: #FFFFFF !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] th {
        background-color: #0e1117 !important;
        color: #00BFFF !important;
        font-weight: 700 !important;
        text-transform: capitalize !important;
    }

    div[data-testid="stDataFrame"] table {
        border: 1px solid #00BFFF !important;
        border-radius: 8px !important;
    }

    div[data-testid="stDataFrame"]::-webkit-scrollbar {
        height: 10px;
        width: 10px;
    }
    div[data-testid="stDataFrame"]::-webkit-scrollbar-thumb {
        background-color: #00BFFF;
        border-radius: 6px;
    }
    div[data-testid="stDataFrame"]::-webkit-scrollbar-thumb:hover {
        background-color: #0099cc;
    }
            /* =============================
   ⚙️ Graph Section Text Fix — Sentiment Distribution
============================= */

/* Graph Title */
[data-testid="stMarkdownContainer"] h5, 
[data-testid="stMarkdownContainer"] h6 {
    color: #ffffff !important; /* white for headings like Sentiment Count Distribution */
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.5);
    font-weight: 700 !important;
}

/* Axis Labels (x-axis + y-axis) */
g.xtick text, 
g.ytick text, 
g.cartesianlayer text {
    fill: #ffffff !important; /* white axis labels */
    font-weight: 600 !important;
}

/* Legend Text (Positive, Negative, etc.) */
.legend text {
    fill: #ffffff !important; /* white legend labels */
    font-weight: 600 !important;
}

/* Bar Label Text (numbers on bars) */
g.barlayer text {
    fill: #ffffff !important; /* make bar value numbers white */
    font-weight: 700 !important;
    text-shadow: 0 0 5px #00BFFF;
}
/* =============================
   ⚙️ Graph Title & X-Axis Label (Better Contrast)
============================= */

/* =============================
   ⚙️ Graph Title & X-Axis Label (Fixed Color)
============================= */

/* Graph Title - Sentiment Count Distribution */
.js-plotly-plot .main-svg text.gtitle,
g.title text,
g.annotation text {
    fill: #80EFFF !important;  /* soft cyan-blue */
    font-weight: 700 !important;
    text-shadow: 0 0 8px rgba(0, 191, 255, 0.6);
}

/* X-Axis Title → "sentiment" in glowing blue */
g.xtitle text,
g.xaxislayer-above > text,
g.xaxis > text {
    fill: #00BFFF !important;  /* Neon blue for axis title */
    font-weight: 700 !important;
    font-size: 18px !important;
    text-shadow: 0 0 10px rgba(0, 191, 255, 0.8);
}

/* X-Axis Tick Labels → (Negative, Positive, Neutral, Irrelevant) in white */
g.xaxislayer-above g.xtick text,
g.xaxis g.tick text,
g.xtick text {
    fill: #FFFFFF !important;  /* Pure white for tick labels */
    font-weight: 600 !important;
    font-size: 16px !important;
}
g.xtitle text,
g.xaxislayer-above > text,
g.xaxis > text {
    fill:  #00BFFF !important;  /* pure white for title */
    font-weight: 700 !important;
    font-size: 18px !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
}
g.ytitle text,
g.yaxislayer-above > text,
g.yaxis > text {
    fill: #00BFFF !important;  /* white title */
    font-weight: 700 !important;
    font-size: 18px !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.4);
}
            /* ============================================
   ⚙️ Fix: Detailed Model Report text always visible (white)
============================================ */
[data-testid="stExpander"] pre,
[data-testid="stExpander"] code,
[data-testid="stMarkdownContainer"] pre,
[data-testid="stMarkdownContainer"] code,
.stMarkdown pre,
.stMarkdown code,
pre,
code {
    color: #FFFFFF !important;              /* Force pure white text */
    background-color: #0e1117 !important;   /* Match dashboard dark background */
    font-weight: 600 !important;
    font-size: 16px !important;
    text-shadow: none !important;
    border: none !important;
}

/* Target nested spans & divs inside expander text */
[data-testid="stExpander"] pre *,
[data-testid="stExpander"] code *,
[data-testid="stMarkdownContainer"] pre *,
[data-testid="stMarkdownContainer"] code *,
pre *,
code * {
    color: #FFFFFF !important;              /* Make sure inner text is also white */
    background: transparent !important;
}

/* Disable any highlight or hover color change */
::selection,
::-moz-selection {
    background: transparent !important;
    color: #FFFFFF !important;
}
/* Make the "Enter a tweet to analyze sentiment" label pure white */
label div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;  /* plain white */
    text-shadow: none !important;  /* remove glow */
    font-weight: 500 !important;
}


            </style>
""", unsafe_allow_html=True)



# ----------------------------------------
# Header
# ----------------------------------------
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:#00BFFF; font-size: 2.6em;'>💬 Twitter Sentiment Analysis Dashboard</h1>
        <p style='color:#aaaaaa; font-size: 1.2em;'>Analyze, visualize, and predict tweet sentiments beautifully 🌍</p>
        <hr style='border: 1px solid #00BFFF; width: 70%; margin:auto;'>
    </div>
""", unsafe_allow_html=True)

# ----------------------------------------
# Sidebar
# ----------------------------------------
st.sidebar.header("⚙️ Dashboard Controls")
uploaded_file = st.sidebar.file_uploader("📁 Upload your Twitter dataset (CSV)", type=['csv'])

# ----------------------------------------
# Data Loading
# ----------------------------------------
if uploaded_file is None:
    st.warning("⚠️ Please upload your dataset to start the analysis.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset Loaded Successfully!")
except Exception as e:
    st.error(f"❌ Error loading file: {e}")
    st.stop()

# ----------------------------------------
# Dataset Overview
# ----------------------------------------
st.subheader("📊 Dataset Overview")
with st.expander("Show Raw Data"):
    st.dataframe(df.head())

st.write("**Shape of Data:**", df.shape)
st.write("**Columns:**", list(df.columns))
st.write("**Missing Values:**")
st.dataframe(df.isnull().sum())

# ----------------------------------------
# Sentiment Distribution
# ----------------------------------------
st.markdown("<h3>📈 Sentiment Distribution</h3>", unsafe_allow_html=True)
sentiment_counts = df['sentiment'].value_counts()

fig = px.bar(
    sentiment_counts,
    x=sentiment_counts.index,
    y=sentiment_counts.values,
    color=sentiment_counts.index,
    color_discrete_sequence=px.colors.sequential.Tealgrn,
    title="Sentiment Count Distribution",
    text_auto=True
)
fig.update_layout(
    plot_bgcolor="#0e1117",
    paper_bgcolor="#0e1117",
    font_color="white"
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------
# WordCloud Section
# ----------------------------------------
st.markdown("<h3>☁️ Word Clouds for Each Sentiment</h3>", unsafe_allow_html=True)
for sentiment in df['sentiment'].unique():
    st.markdown(f"#### {sentiment}")
    text = " ".join(df[df['sentiment'] == sentiment]['tweet'].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='#0e1117', colormap='cool').generate(text)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

# ----------------------------------------
# Model Training
# ----------------------------------------
st.markdown("<h3>🤖 Train & Evaluate Sentiment Model</h3>", unsafe_allow_html=True)
df = df.dropna(subset=['tweet'])
df = df[df['tweet'].str.strip() != ""]

X = df['tweet'].astype(str)
y = df['sentiment']

vectorizer = CountVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
st.markdown(f"<h4 style='color:#00FF7F;'>🎯 Model Accuracy: {accuracy*100:.2f}%</h4>", unsafe_allow_html=True)

# Classification Report
with st.expander("📋 Detailed Model Report"):
    st.code(classification_report(y_test, pred))

# Confusion Matrix
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='cool', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
st.pyplot(fig)

# ----------------------------------------
# User Prediction
# ----------------------------------------
st.markdown("<h3>💬 Test Your Own Tweet</h3>", unsafe_allow_html=True)
user_input = st.text_area("✍️ Enter a tweet to analyze sentiment:")

if st.button("🔍 Predict Sentiment"):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', user_input.lower())
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)[0]
    polarity = TextBlob(user_input).sentiment.polarity

    st.markdown(f"<h4>Predicted Sentiment: <span style='color:#00BFFF;'>{prediction}</span></h4>", unsafe_allow_html=True)
    st.write(f"**Polarity Score:** `{polarity:.2f}`")

    if polarity > 0:
        st.success("😀 Positive Sentiment Detected!")
    elif polarity < 0:
        st.error("😞 Negative Sentiment Detected!")
    else:
        st.info("😐 Neutral Sentiment Detected!")

# ----------------------------------------
# Footer
# ----------------------------------------
st.markdown("""
    <hr>
    <div style='text-align: center; color:#aaaaaa; font-size:0.9em;'>
        <p>✨ Developed  by <b style='color:#aaaaaa;'>Dharshini and jayasri...</b> | Powered by Streamlit, Plotly & NLP 🚀</p>
    </div>
""", unsafe_allow_html=True)
