
import streamlit as st
from PIL import Image
from transformers import pipeline
import pandas as pd

st.set_page_config(
    page_title="MoodFlix",
    page_icon="🎬",
    layout="centered"
)

st.title("🎬 MoodFlix")
st.subheader("Facial Expression Movie Recommendation")
st.write(
    "Upload a face photo to estimate its visible expression "
    "and discover movies for your movie night."
)

# Movie recommendations for each detected expression
MOVIES = {
    "happy": [
        ("The Lion King", "Animation / Adventure"),
        ("Paddington 2", "Comedy / Family"),
        ("Inside Out", "Animation / Comedy"),
        ("Zootopia", "Animation / Mystery"),
    ],
    "sad": [
        ("The Pursuit of Happyness", "Drama"),
        ("Wonder", "Drama / Family"),
        ("The Secret Life of Walter Mitty", "Adventure"),
        ("Finding Nemo", "Animation / Adventure"),
    ],
    "angry": [
        ("The Avengers", "Action / Adventure"),
        ("Spider-Man: Into the Spider-Verse", "Animation / Action"),
        ("Kung Fu Panda", "Animation / Comedy"),
        ("Top Gun: Maverick", "Action / Drama"),
    ],
    "fear": [
        ("Finding Nemo", "Animation / Adventure"),
        ("Paddington 2", "Comedy / Family"),
        ("Toy Story", "Animation / Adventure"),
        ("The Super Mario Bros. Movie", "Animation / Comedy"),
    ],
    "surprise": [
        ("Inception", "Sci-Fi / Thriller"),
        ("Interstellar", "Sci-Fi / Drama"),
        ("Knives Out", "Mystery / Comedy"),
        ("Jumanji: Welcome to the Jungle", "Adventure / Comedy"),
    ],
    "disgust": [
        ("Ratatouille", "Animation / Comedy"),
        ("Toy Story", "Animation / Adventure"),
        ("Finding Nemo", "Animation / Adventure"),
        ("The Intern", "Comedy / Drama"),
    ],
    "neutral": [
        ("The Martian", "Sci-Fi / Adventure"),
        ("The Grand Budapest Hotel", "Comedy / Adventure"),
        ("Harry Potter and the Sorcerer's Stone", "Fantasy"),
        ("How to Train Your Dragon", "Animation / Adventure"),
    ],
}

@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression"
    )

uploaded_file = st.file_uploader(
    "📷 Upload a face photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Your uploaded photo",
                 use_container_width=True)

        if st.button("🔍 Detect Expression", type="primary"):
            with st.spinner(
                "Loading the AI model and analyzing the image..."
            ):
                classifier = load_model()
                predictions = classifier(image)

            if not predictions:
                st.error("The model did not return a prediction.")
                st.stop()

            # Get the highest-scoring expression
            best = predictions[0]
            expression = best["label"].strip().lower()
            confidence = float(best["score"])

            st.success(
                f"Estimated expression: {expression.title()}"
            )
            st.metric(
                "Model confidence",
                f"{confidence:.1%}"
            )

            st.caption(
                "This is an AI estimate of visible facial expression, "
                "not a reliable measure of someone's true feelings."
            )

            # Recommend movies for the predicted label
            if expression in MOVIES:
                recommendations = MOVIES[expression]
            else:
                recommendations = MOVIES["neutral"]
                st.info(
                    "This expression label is not in the movie list, "
                    "so general recommendations are shown."
                )

            st.header("🍿 Recommended Movies")

            for number, (title, genre) in enumerate(
                recommendations, start=1
            ):
                st.write(f"**{number}. {title}**")
                st.caption(f"Genre: {genre}")

            with st.expander("See all model predictions"):
                for item in predictions[:7]:
                    st.write(
                        f"{item['label']}: {item['score']:.1%}"
                    )

    except Exception as error:
        st.error(
            "The model could not run. Check your internet connection "
            "and installed dependencies, then try again."
        )
        st.code(str(error))

st.divider()
st.caption("MoodFlix | Smart Movie Recommendation Mini Project")
