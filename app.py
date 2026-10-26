

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
from transformers import pipeline

# -----------------------------
# MoodFlix configuration
# -----------------------------
st.set_page_config(
    page_title="MoodFlix",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 MoodFlix")
st.subheader("Smart Movie Recommendation System")
st.write(
    "Discover movies using a visible facial-expression estimate "
    "or choose an expression yourself."
)
st.caption(
    "Facial expressions cannot reliably reveal true feelings. "
    "Images are analyzed only when you choose to analyze them."
)

CSV_PATH = Path(__file__).parent / "moodflix_100_movies_dataset.csv"


# -----------------------------
# Load the 100-movie dataset
# -----------------------------
@st.cache_data
def load_movies():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Upload "
            "moodflix_100_movies_dataset.csv beside app.py."
        )

    df = pd.read_csv(CSV_PATH)

    required = {"title", "genres", "expression"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing CSV columns: " + ", ".join(sorted(missing))
        )

    df = df.dropna(subset=["title", "expression"]).copy()
    df["title"] = df["title"].astype(str).str.strip()
    df["expression"] = df["expression"].astype(str).str.lower().str.strip()
    df["genres"] = df["genres"].fillna("Not listed").astype(str)

    return df


try:
    movies_df = load_movies()
except Exception as error:
    st.error(f"Unable to load the movie dataset: {error}")
    st.stop()


st.success(f"🎞️ Dataset loaded: {len(movies_df)} movies")


# -----------------------------
# Normalize expression names
# -----------------------------
def normalize_expression(label):
    label = str(label).lower().strip()

    aliases = {
        "happiness": "happy",
        "joy": "happy",
        "sadness": "sad",
        "anger": "angry",
        "fearful": "fear",
        "scared": "fear",
        "surprised": "surprise",
        "disgusted": "disgust",
    }

    return aliases.get(label, label)


# -----------------------------
# Load facial-expression model
# -----------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression",
        device=-1
    )


# -----------------------------
# Camera or photo upload
# -----------------------------
st.header("📸 Step 1: Choose a photo")

camera_tab, upload_tab = st.tabs(
    ["Use camera", "Upload photo"]
)

image_file = None

with camera_tab:
    camera_photo = st.camera_input("Take a photo")

    if camera_photo is not None:
        image_file = camera_photo

with upload_tab:
    uploaded_photo = st.file_uploader(
        "Choose a face photo",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_photo is not None:
        image_file = uploaded_photo


# -----------------------------
# Choose recommendation method
# -----------------------------
st.header("🎭 Step 2: Choose an expression")

method = st.radio(
    "How would you like to choose?",
    [
        "Analyze photo",
        "Choose manually"
    ],
    horizontal=True
)

expression = None

if method == "Analyze photo":
    if image_file is not None:
        try:
            image = Image.open(image_file).convert("RGB")
            st.image(image, caption="Selected photo", width=280)

            if st.button("🔍 Analyze expression"):
                with st.spinner(
                    "Loading model and estimating expression..."
                ):
                    classifier = load_model()
                    predictions = classifier(image)

                if predictions:
                    result = predictions[0]
                    st.session_state["moodflix_expression"] = (
                        normalize_expression(result["label"])
                    )
                    st.session_state["moodflix_score"] = (
                        result["score"]
                    )

        except Exception:
            st.error(
                "Photo analysis failed. Try another photo "
                "or select an expression manually."
            )

    if "moodflix_expression" in st.session_state:
        expression = st.session_state["moodflix_expression"]
        score = st.session_state.get("moodflix_score", 0.0)

        st.info(f"Estimated visible expression: **{expression}**")
        st.caption(
            f"Model score: {score:.1%}. This score is not "
            "a measure of how accurately your true mood is known."
        )

else:
    available_expressions = sorted(
        movies_df["expression"].dropna().unique().tolist()
    )

    if available_expressions:
        expression = st.selectbox(
            "Select an expression from your dataset",
            available_expressions
        )


# -----------------------------
# Recommend movies from CSV
# -----------------------------
st.header("🍿 Step 3: Movie recommendations")

if expression:
    target = normalize_expression(expression)

    matching = movies_df[
        movies_df["expression"].map(normalize_expression) == target
    ]

    if matching.empty:
        st.warning(
            f"No movies are tagged '{expression}'. "
            "Here are some movies from your dataset instead."
        )
        matching = movies_df

    st.write(f"Found **{len(matching)}** matching movies.")

    for _, movie in matching.iterrows():
        with st.container(border=True):
            st.subheader(f"🎬 {movie['title']}")

            if "year" in movies_df.columns:
                year = movie["year"]
                if pd.notna(year):
                    st.write(f"**Year:** {year}")

            st.write(f"**Genres:** {movie['genres']}")

            if "category" in movies_df.columns:
                category = movie["category"]
                if pd.notna(category):
                    st.write(f"**Category:** {category}")

else:
    st.info(
        "Take or upload a photo and tap Analyze expression, "
        "or choose an expression manually to see movie recommendations."
    )


# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "MoodFlix | Student mini-project | "
    "Recommendations use your CSV dataset."
)
