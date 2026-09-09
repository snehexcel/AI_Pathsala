import dspy
from pydantic import BaseModel, Field
from chroma import qdrant
from config import MISTRAL_API_KEY


# Mistral LLM
llm = dspy.LM(
    model="mistral-small-latest",
    api_key=MISTRAL_API_KEY,
    api_base="https://api.mistral.ai/v1",
    num_retries=0
)


# =========================
# CHATBOT
# =========================

class QuerySignature(dspy.Signature):
    """
    Provide complete and to-the-point answers to student queries regarding
    their subjects, including both theoretical questions and numerical
    problems, using content from textbooks.

    You are great in mathematics, so show proper steps to solve numericals.
    """

    context = dspy.InputField(
        desc="Relevant facts from textbooks"
    )

    question: str = dspy.InputField(
        desc="Student's question, either theoretical or numerical"
    )

    answer: str = dspy.OutputField(
        desc="Complete and to-the-point answer"
    )


class ChatbotRAG(dspy.Module):

    def __init__(self):
        super().__init__()

        self.generate_answer = dspy.Predict(
            signature=QuerySignature
        )

    def forward(self, question):

        # Retrieve relevant content from Qdrant
        context = qdrant.search(
            query=question,
            search_type="similarity_score_threshold"
        )

        # Use Mistral only for this request
        with dspy.context(lm=llm):

            prediction = self.generate_answer(
                context=context,
                question=question
            )

        return dspy.Prediction(
            context=context,
            answer=prediction.answer
        )


# =========================
# QUIZ
# =========================

class QuizInput(BaseModel):

    topic: str = Field(
        description="The topic for the quiz"
    )

    context: list[str] = Field(
        description="Relevant context from Qdrant"
    )


class QuizOption(BaseModel):

    option: str = Field(
        description="A possible answer option"
    )


class QuizOutput(BaseModel):

    question: str = Field(
        description="The generated quiz question"
    )

    options: list[QuizOption] = Field(
        description="Exactly four answer options"
    )

    correct_option: int = Field(
        ge=0,
        le=3,
        description="Index of the correct answer option"
    )


class QuizSignature(dspy.Signature):

    """
    Generate a quiz question on a user-provided topic
    with four answer options and identify the correct option.
    """

    input: QuizInput = dspy.InputField()

    output: QuizOutput = dspy.OutputField()


class QuizRAG(dspy.Module):

    def __init__(self):
        super().__init__()

        self.generate_quiz = dspy.ChainOfThought(
            signature=QuizSignature
        )

    def forward(self, quiz_text):

        # Retrieve relevant content from Qdrant
        context = qdrant.search(
            query=quiz_text,
            search_type="similarity_score_threshold"
        )

        # Convert retrieved documents into plain text
        context_text = []

        for doc in context:
            context_text.append(
                str(doc.page_content)
            )

        # Create structured quiz input
        quiz_input = QuizInput(
            topic=str(quiz_text),
            context=context_text
        )

        # Use Mistral only for this request
        with dspy.context(lm=llm):

            prediction = self.generate_quiz(
                input=quiz_input
            )

        return prediction
