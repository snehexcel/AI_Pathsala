import dspy 
import streamlit as st
import os
from pydantic import BaseModel, Field

# IMPORT QDRANT FROM YOUR OTHER FILE
# Make sure your other file is named 'chroma.py' and it has the 'qdrant' variable!
from chroma import qdrant 

# --- 1. SECURE KEY HANDLING ---
try:
    # Try getting key from Streamlit Cloud Secrets
    google_api_key = st.secrets["GOOGLE_API_KEY"]
except:
    # Fallback to local .env
    from dotenv import load_dotenv
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")

# --- 2. CONFIGURE DSPY WITH GEMINI ---
# using gemini-1.5-flash as it is most stable for free tier
llm = dspy.LM("gemini/gemini-1.5-flash", api_key=google_api_key)
dspy.settings.configure(lm=llm)

# --- 3. SIGNATURES ---
class QuerySignature(dspy.Signature):
    '''
    Provide complete and to-the-point answers to student queries regarding their subjects, 
    including both theoretical questions and numerical problems, using content from textbooks.
    *You are great in mathematics so show proper steps to solve numericals*
    '''
    context = dspy.InputField(desc="may contain relevant facts from textbooks")
    question: str = dspy.InputField(desc="Student's question, either theoretical or numerical")
    answer: str = dspy.OutputField(desc="Complete and to-the-point answer")

class QuizInput(BaseModel):
    topic: str = Field(description="The topic for the quiz")
    context: list[str] = Field(description="Relevant context from ChromaDB")

class QuizOption(BaseModel):
    option: str = Field(description="A possible answer option")

class QuizOutput(BaseModel):
    question: str = Field(description="The generated quiz question")
    options: list[QuizOption] = Field(description="The list of answer options")
    correct_option: int = Field(ge=0, le=3, description="The index of the correct answer option")

class QuizSignature(dspy.Signature):
    """Generate a quiz question on a user-provided topic with 4 answer options."""
    input: QuizInput = dspy.InputField()
    output: QuizOutput = dspy.OutputField()


# --- 4. MODULES ---
class ChatbotRAG(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(signature=QuerySignature)

    def forward(self, question):
        # 1. Search Qdrant
        results = qdrant.similarity_search(
            query=question,
            k=4 
        )
        
        # 2. Convert LangChain Docs to String for DSPy
        # DSPy crashes if you give it raw Document objects
        context_text = "\n\n".join([doc.page_content for doc in results])
        
        # 3. Generate Answer
        prediction = self.generate_answer(context=context_text, question=question)
        return dspy.Prediction(context=results, answer=prediction.answer)

class QuizRAG(dspy.Module):
    def __init__(self):
        super().__init__() 
        self.generate_quiz = dspy.ChainOfThought(QuizSignature)

    def forward(self, quiz_text):
        # 1. Search Qdrant
        results = qdrant.similarity_search(
            query=quiz_text,
            k=4
        )
        
        # 2. Extract text list for Pydantic
        context_text = [doc.page_content for doc in results]
        
        # 3. Create Input Object
        quiz_input = QuizInput(topic=str(quiz_text), context=context_text)
        
        # 4. Generate Quiz
        prediction = self.generate_quiz(input=quiz_input)
        return prediction
