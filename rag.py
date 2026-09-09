import requests
import json
import streamlit as st

from pydantic import BaseModel, Field
from chroma import qdrant


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-2.5-flash:generateContent"
)


# ============================================================
# GEMINI HELPER
# ============================================================

def call_gemini(prompt, temperature=0.3, max_output_tokens=1200):

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY,
        },
        json={
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens,
            },
        },
        timeout=60,
    )

    if response.status_code != 200:
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get(
                "message",
                "Unknown Gemini API error"
            )
        except Exception:
            error_message = response.text

        raise RuntimeError(
            f"Gemini API error ({response.status_code}): {error_message}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(
            "Gemini returned an unexpected response."
        )


# ============================================================
# QDRANT RETRIEVAL
# ============================================================

def retrieve_context(query):

    try:

        documents = qdrant.search(
            query=query,
            search_type="similarity_score_threshold"
        )

        context = []

        for doc in documents:
            try:
                text = str(doc.page_content).strip()

                if text:
                    context.append(text)

            except Exception:
                continue

        return context

    except Exception:
        return []


# ============================================================
# CHATBOT
# ============================================================

class ChatbotRAG:

    def forward(self, question):

        question = str(question).strip()

        if not question:
            return {
                "context": [],
                "answer": "Please enter a question."
            }

        context = retrieve_context(question)

        if context:

            context_text = "\n\n".join(context)

        else:

            context_text = (
                "No relevant textbook content was found in the "
                "knowledge base."
            )

        prompt = f"""
You are the AIPathshala educational assistant.

Answer the student's question clearly, accurately and concisely.

Use the textbook context below when it is relevant.

If the question is a numerical problem:
- Show the formula.
- Show the calculation steps.
- Give the final answer clearly.

If the textbook context does not contain the answer, use your
general knowledge rather than inventing information.

TEXTBOOK CONTEXT:
{context_text}

STUDENT QUESTION:
{question}

Provide a helpful student-friendly answer.
"""

        answer = call_gemini(
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=1200
        )

        return {
            "context": context,
            "answer": answer
        }


# ============================================================
# QUIZ MODELS
# ============================================================

class QuizOption(BaseModel):

    option: str


class QuizOutput(BaseModel):

    question: str

    options: list[QuizOption]

    correct_option: int = Field(
        ge=0,
        le=3
    )


# ============================================================
# QUIZ
# ============================================================

class QuizRAG:

    def forward(self, quiz_text):

        quiz_text = str(quiz_text).strip()

        context = retrieve_context(quiz_text)

        if context:

            context_text = "\n\n".join(context)

        else:

            context_text = (
                "No textbook context was found. "
                "Use reliable general knowledge."
            )

        prompt = f"""
You are an educational quiz generator for AIPathshala.

Create exactly ONE multiple-choice question about:

{quiz_text}

Use this textbook context when relevant:

{context_text}

Requirements:

1. Create exactly four options.
2. Only one option must be correct.
3. correct_option must be the zero-based index:
   0, 1, 2, or 3.
4. Keep the question suitable for a B.Tech/CSE student.
5. Return ONLY valid JSON.
6. Do not include Markdown.
7. Do not include explanations outside the JSON.

Return exactly this structure:

{{
    "question": "Question text",
    "options": [
        {{"option": "Option 1"}},
        {{"option": "Option 2"}},
        {{"option": "Option 3"}},
        {{"option": "Option 4"}}
    ],
    "correct_option": 0
}}
"""

        raw_response = call_gemini(
            prompt=prompt,
            temperature=0.4,
            max_output_tokens=700
        )

        # Remove possible Markdown fences
        raw_response = raw_response.strip()

        if raw_response.startswith("```"):
            raw_response = raw_response.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        try:

            data = json.loads(raw_response)

            quiz = QuizOutput.model_validate(data)

            if len(quiz.options) != 4:
                raise ValueError(
                    "Gemini did not return exactly four options."
                )

            return type(
                "QuizPrediction",
                (),
                {
                    "output": quiz
                }
            )()

        except Exception as e:

            raise RuntimeError(
                f"Could not parse Gemini quiz response: {e}"
            )
