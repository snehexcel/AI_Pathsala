import streamlit as st
import os
import hashlib
from io import BytesIO

import nltk
from icrawler.builtin import GoogleImageCrawler
from pytube import Search
from elevenlabs.client import ElevenLabs

from rag import ChatbotRAG
from web_content import *
from config import ELEVENLABS_API_KEY


# ============================================================
# PAGE UI
# ============================================================

st.markdown(
    hero_logo,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        sidebar_logo,
        unsafe_allow_html=True
    )


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "last_audio_id" not in st.session_state:
    st.session_state.last_audio_id = None

if "media_question" not in st.session_state:
    st.session_state.media_question = None

if "images" not in st.session_state:
    st.session_state.images = []

if "video_url" not in st.session_state:
    st.session_state.video_url = None


# ============================================================
# ELEVENLABS
# ============================================================

@st.cache_resource
def get_elevenlabs_client():

    return ElevenLabs(
        api_key=ELEVENLABS_API_KEY
    )


client = get_elevenlabs_client()


# ============================================================
# NLTK
# ============================================================

@st.cache_resource
def setup_nltk():

    nltk.download(
        "averaged_perceptron_tagger",
        quiet=True
    )

    nltk.download(
        "punkt",
        quiet=True
    )


setup_nltk()


# ============================================================
# GOOGLE IMAGE CRAWLER
# ============================================================

@st.cache_resource
def get_google_crawler():

    return GoogleImageCrawler(
        storage={
            "root_dir": "Images"
        }
    )


google_crawler = get_google_crawler()


# ============================================================
# GET AI RESPONSE
# ============================================================

def get_response(question):

    return ChatbotRAG().forward(
        question=question
    )


# ============================================================
# PROCESS QUESTION
# ============================================================

def process_question(question):

    if not question:
        return

    question = question.strip()

    if not question:
        return

    try:

        with st.spinner(
            "Waiting for response..."
        ):

            response = get_response(
                question
            )

        # Store question + answer
        st.session_state.history.append(
            (
                question,
                response.answer
            )
        )

        # Mark question for media loading
        st.session_state.media_question = (
            question
        )

        # Clear previous media
        st.session_state.images = []

        st.session_state.video_url = None

    except Exception as e:

        st.error(
            "Unable to generate an answer."
        )

        st.error(
            str(e)
        )


# ============================================================
# LOAD IMAGES + VIDEO
# ============================================================

def load_media(question, answer):

    if (
        st.session_state.media_question
        != question
    ):
        return

    # Already loaded
    if (
        st.session_state.images
        or st.session_state.video_url
        is not None
    ):
        return


    # ========================================================
    # IMAGES
    # ========================================================

    try:

        image_folder = "Images"

        os.makedirs(
            image_folder,
            exist_ok=True
        )

        # Remove old images
        for filename in os.listdir(
            image_folder
        ):

            file_path = os.path.join(
                image_folder,
                filename
            )

            if filename.lower().endswith(
                (
                    "png",
                    "jpg",
                    "jpeg"
                )
            ):

                try:

                    os.remove(
                        file_path
                    )

                except Exception:
                    pass


        # Search relevant images
        google_crawler.crawl(

            keyword=(
                "relevant educational "
                "diagram or picture "
                "from NCERT textbook - "
                f"Question: {question}, "
                f"Answer: {answer}"
            ),

            max_num=5
        )


        images = []

        if os.path.exists(
            image_folder
        ):

            for filename in os.listdir(
                image_folder
            ):

                if filename.lower().endswith(
                    (
                        "png",
                        "jpg",
                        "jpeg"
                    )
                ):

                    images.append(
                        os.path.join(
                            image_folder,
                            filename
                        )
                    )


        st.session_state.images = (
            images
        )


    except Exception:

        st.session_state.images = []


    # ========================================================
    # VIDEO
    # ========================================================

    try:

        search_video = Search(
            question
        )

        results = (
            search_video.results
        )

        if results:

            st.session_state.video_url = (
                results[0].watch_url
            )

        else:

            st.session_state.video_url = (
                None
            )


    except Exception:

        st.session_state.video_url = None


# ============================================================
# LAYOUT
# ============================================================

s1, s2 = st.columns(
    [3, 1]
)


# ============================================================
# LEFT COLUMN — CHATBOT
# ============================================================

with s1:

    st.header(
        "Ask Me 💭"
    )

    option = st.selectbox(

        label="Select the input option",

        options=[
            "Speak",
            "Write"
        ]
    )


    # ========================================================
    # SPEAK
    # ========================================================

    if option == "Speak":

        ask_question = st.audio_input(
            "Ask a question:"
        )


        if ask_question:

            audio_bytes = (
                ask_question.getvalue()
            )


            # Create unique ID
            audio_id = hashlib.md5(
                audio_bytes
            ).hexdigest()


            # Prevent duplicate processing
            if (
                audio_id
                != st.session_state.last_audio_id
            ):

                st.session_state.last_audio_id = (
                    audio_id
                )


                try:

                    audio_data = BytesIO(
                        audio_bytes
                    )


                    transcription = (
                        client
                        .speech_to_text
                        .convert(

                            file=audio_data,

                            model_id="scribe_v1",

                            tag_audio_events=True,

                            language_code="eng",

                            diarize=True
                        )
                    )


                    question = (
                        transcription.text
                    )


                    process_question(
                        question
                    )


                except Exception as e:

                    st.error(
                        "Speech processing error:"
                    )

                    st.error(
                        str(e)
                    )


    # ========================================================
    # WRITE
    # ========================================================

    if option == "Write":

        question = st.text_input(
            label="Write your query here"
        )


        if st.button(
            "Ask",
            type="primary"
        ):

            if question.strip():

                process_question(
                    question
                )

            else:

                st.warning(
                    "Please enter a question."
                )


    # ========================================================
    # CHAT HISTORY
    # ========================================================

    if st.session_state.history:

        # Older messages
        for i in range(
            len(
                st.session_state.history
            ) - 1
        ):

            q, r = (
                st.session_state.history[i]
            )


            with st.chat_message(
                "user"
            ):

                st.write(q)


            with st.chat_message(
                "assistant"
            ):

                st.write(r)


        # Latest message
        q, r = (
            st.session_state.history[-1]
        )


        with st.chat_message(
            "user"
        ):

            st.write(q)


        with st.chat_message(
            "assistant"
        ):

            st.write(r)


# ============================================================
# RIGHT COLUMN — MEDIA
# ============================================================

with s2:

    if st.session_state.history:

        q, r = (
            st.session_state.history[-1]
        )


        # Load media
        load_media(
            q,
            r
        )


        # ====================================================
        # IMAGES
        # ====================================================

        st.header(
            "Images"
        )


        images = (
            st.session_state.images
        )


        if images:

            with st.container(
                border=True,
                height=400
            ):

                st.image(

                    images,

                    caption=[
                        os.path.basename(
                            img
                        )
                        for img in images
                    ],

                    use_container_width=True
                )

        else:

            st.caption(
                "No relevant images found."
            )


        # ====================================================
        # VIDEO
        # ====================================================

        st.header(
            "Video Explanation"
        )


        if st.session_state.video_url:

            st.video(
                st.session_state.video_url
            )

        else:

            st.caption(
                "No video explanation found."
            )
