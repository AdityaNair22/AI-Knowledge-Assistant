"""
emotion_model.py (optional)
----------------------------
Detects emotional tone and adds empathy prefix to answers.
Uses Groq (free) instead of OpenAI.
"""

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def detect_emotion(text: str) -> str:
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    prompt = ChatPromptTemplate.from_template("""
Classify the emotional tone of this message into ONE word only.
Choose from: frustrated, curious, confused, neutral

Message: {text}

Reply with ONE word only:
""")

    result = (prompt | llm).invoke({"text": text})
    tone   = result.content.strip().lower()

    if tone not in ["frustrated", "curious", "confused", "neutral"]:
        return "neutral"
    return tone


def get_empathy_prefix(tone: str) -> str:
    prefixes = {
        "frustrated": "I understand this can be tricky. Let me help clarify — ",
        "curious":    "Great question! ",
        "confused":   "No worries, let me explain this clearly — ",
        "neutral":    ""
    }
    return prefixes.get(tone, "")