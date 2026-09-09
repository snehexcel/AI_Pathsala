import json
import requests
import streamlit as st

from dataclasses import dataclass
from pydantic import BaseModel, Field

from chroma import qdrant


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"].strip()

# TEMPORARY DIAGNOSTIC
# Shows only the last 4 characters of the loaded key.
# Never exposes the complete API key.
st.sidebar.caption(
    f"Gemini key loaded: ****{GOOGLE_API_KEY[-4:]}"
)

GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-3.5-flash-lite:generateContent"
)


# ============================================================
# CHAT RESPONSE
# ============================================================

@dataclass
class ChatResponse:
    context: list
    answer: str


# ============================================================
# CALL GEMINI
# ============================================================

def call_gemini(
    prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 1200
) -> str:

    response = requests.post(
        GEMINI_URL,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GOOGLE_API_KEY,
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_output_tokens
            }
        },
        timeout=60
    )

    # ========================================================
    # SUCCESS
    # ========================================================

    if response.status_code == 200:

        data = response.json()

        try:

            answer = (
                data["candidates"][0]
                ["content"]["parts"][0]["text"]
            )

            return answer.strip()

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            raise RuntimeError(
                "Gemini returned an unexpected response."
            )

    # ========================================================
    # ERROR
    # ========================================================

    try:

        error_data = response.json()

        error = error_data.get(
            "error",
            {}
        )

        error_message = error.get(
            "message",
            "Unknown Gemini API error."
        )

        error_status = error.get(
            "status",
            "UNKNOWN"
        )

    except Exception:

        error_message = response.text
        error_status = "UNKNOWN"

    raise RuntimeError(
        f"Gemini API error "
        f"(HTTP {response.status_code}, "
        f"{error_status}): "
        f"{error_message}"
    )


# ============================================================
# QDRANT CONTEXT RETRIEVAL
# ============================================================

def retrieve_context(query: str) -> list:

    try:

        documents = qdrant.search(
            query=query,
            search_type="similarity_score_threshold"
        )

    except Exception:

        return []

    context = []

    for document in documents:

        try:

            text = str(
                document.page_content
            ).strip()

            if text:
                context.append(text)

        except Exception:

            continue

    return context


# ============================================================
# CHATBOT RAG
# ============================================================

class ChatbotRAG:

    def forward(self, question):

        question = str(
            question
        ).strip()

        if not question:

            return ChatResponse(
                context=[],
                answer="Please enter a question."
            )

        # ----------------------------------------------------
        # RETRIEVE TEXTBOOK CONTEXT
        # ----------------------------------------------------

        context = retrieve_context(
            question
        )

        if context:

            context_text = "\n\n".join(
                context
            )

        else:

            context_text = (
                "No relevant textbook content "
                "was found in the knowledge base."
            )

        # ----------------------------------------------------
        # CHATBOT PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are AIPathshala, an AI educational
assistant for students.

Answer the student's question clearly,
accurately, and in a student-friendly way.

Use the textbook context below when it
is relevant.

If the question is a numerical problem:

1. Write the formula.
2. Show the calculation steps.
3. Explain the reasoning.
4. Give the final answer clearly.

If the textbook context does not contain
enough information, use reliable general
knowledge.

Do not invent textbook information.

Do not mention these instructions.

================ TEXTBOOK CONTEXT ================

{context_text}

================ STUDENT QUESTION ================

{question}

================ ANSWER ================

Give a complete but concise answer.
"""

        # ----------------------------------------------------
        # GEMINI GENERATION
        # ----------------------------------------------------

        answer = call_gemini(
            prompt=prompt,
            temperature=0.2,
            max_output_tokens=1200
        )

        return ChatResponse(
            context=context,
            answer=answer
        )


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


@dataclass
class QuizPrediction:
    output: QuizOutput


# ============================================================
# QUIZ RAG
# ============================================================

class QuizRAG:

    def forward(self, quiz_text):

        quiz_text = str(
            quiz_text
        ).strip()

        if not quiz_text:

            raise RuntimeError(
                "Please enter a quiz topic."
            )

        # ----------------------------------------------------
        # RETRIEVE CONTEXT
        # ----------------------------------------------------

        context = retrieve_context(
            quiz_text
        )

        if context:

            context_text = "\n\n".join(
                context
            )

        else:

            context_text = (
                "No relevant textbook content "
                "was found. Use reliable general knowledge."
            )

        # ----------------------------------------------------
        # QUIZ PROMPT
        # ----------------------------------------------------

        prompt = f"""
You are the quiz generator for AIPathshala.

Create exactly ONE multiple-choice question
about this topic:

{quiz_text}

Use the textbook context when relevant.

================ TEXTBOOK CONTEXT ================

{context_text}

================ REQUIREMENTS ================

Create exactly four answer options.

Only one option must be correct.

The correct_option value must be the
zero-based index:

0 = first option
1 = second option
2 = third option
3 = fourth option

Make the question educational and accurate.

Keep it suitable for a B.Tech/CSE student.

Return ONLY valid JSON.

Do not use Markdown.

Do not add any explanation outside the JSON.

Return exactly:

{{
    "question": "Your question",
    "options": [
        {{"option": "Option 1"}},
        {{"option": "Option 2"}},
        {{"option": "Option 3"}},
        {{"option": "Option 4"}}
    ],
    "correct_option": 0
}}
"""

        # ----------------------------------------------------
        # GENERATE QUIZ
        # ----------------------------------------------------

        raw_response = call_gemini(
            prompt=prompt,
            temperature=0.3,
            max_output_tokens=700
        )

        raw_response = raw_response.strip()

        # ----------------------------------------------------
        # REMOVE MARKDOWN CODE FENCES
        # ----------------------------------------------------

        if raw_response.startswith("```"):

            raw_response = (
                raw_response
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                raw_response
            )

            quiz = QuizOutput.model_validate(
                data
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not parse Gemini quiz response: {e}"
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        if len(quiz.options) != 4:

            raise RuntimeError(
                "Gemini did not return exactly "
                "four answer options."
            )

        if not (
            0 <= quiz.correct_option <= 3
        ):

            raise RuntimeError(
                "Gemini returned an invalid "
                "correct option."
            )

        return QuizPrediction(
            output=quiz
        )
