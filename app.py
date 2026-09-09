import streamlit as st
import requests
from web_content import *

st.set_page_config(
    layout="wide",
    page_title="AIPathsala",
    page_icon=":books:"
)

st.markdown(hero_logo, unsafe_allow_html=True)

with st.sidebar:
    st.markdown(sidebar_logo, unsafe_allow_html=True)


st.title("Mistral API Test")

if st.button("Test Mistral API", type="primary"):

    api_key = st.secrets["MISTRAL_API_KEY"]

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistral-small-latest",
                "messages": [
                    {
                        "role": "user",
                        "content": "What is an electromagnet? Answer in one sentence."
                    }
                ],
            },
            timeout=30,
        )

        st.write("HTTP Status:", response.status_code)

        if response.status_code == 200:
            data = response.json()
            st.success("Mistral API is working!")
            st.write(data["choices"][0]["message"]["content"])

        else:
            st.error("Mistral API request failed.")
            st.code(response.text)

    except Exception as e:
        st.error(f"Connection error: {e}")
