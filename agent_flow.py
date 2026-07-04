"""
agent_flow.py
--------------
Decision Engine using LangGraph 1.2.5 + langchain-core 1.4.7
Uses Groq (free LLM API) instead of OpenAI for chat completions.
"""

from typing import TypedDict, Literal, List
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:  str
    route:     str
    context:   List[Document]
    answer:    str
    memory:    List[dict]


# ── LLM (Groq - free, fast) ───────────────────────────────────────────────────

def get_llm():
    return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)


# ── Node 1: Router ────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> AgentState:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_template("""
You are a routing assistant. Given a user question, decide the best route.

Routes:
- "rag"     : specific question answerable from uploaded PDFs or notes
- "general" : general knowledge question, no documents needed
- "clarify" : question is too vague to answer properly

Question: {question}

Reply with ONLY one word: rag, general, or clarify
""")

    result = (prompt | llm).invoke({"question": state["question"]})
    route  = result.content.strip().lower()

    if route not in ["rag", "general", "clarify"]:
        route = "general"

    return {**state, "route": route}


# ── Node 2: RAG ───────────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    from rag_pipeline import get_retriever
    retriever    = get_retriever()
    llm          = get_llm()
    question     = state["question"]

    docs         = retriever.invoke(question) if retriever else []
    context_text = "\n\n".join([d.page_content for d in docs])

    memory_text  = ""
    if state.get("memory"):
        for m in state["memory"][-3:]:
            memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Knowledge Assistant.
Answer the question directly using ONLY the document context below.
Do NOT add phrases like "No worries" or "Great question".
Do NOT describe what the document is about — just answer directly.
If asked to summarise, write a clear structured summary of all the context provided.
If the answer is not in the context, say "I could not find that in the uploaded documents."

Previous conversation:
{memory}

Document context:
{context}

Question: {question}

Answer:
""")

    result = (prompt | llm).invoke({
        "question": question,
        "context":  context_text,
        "memory":   memory_text
    })

    return {**state, "answer": result.content, "context": docs}


# ── Node 3: General ───────────────────────────────────────────────────────────

def general_node(state: AgentState) -> AgentState:
    llm      = get_llm()
    question = state["question"]

    memory_text = ""
    if state.get("memory"):
        for m in state["memory"][-3:]:
            memory_text += f"User: {m['question']}\nAssistant: {m['answer']}\n"

    prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Knowledge Assistant. Answer using your general knowledge.

Previous conversation:
{memory}

Question: {question}

Answer:
""")

    result = (prompt | llm).invoke({
        "question": question,
        "memory":   memory_text
    })

    return {**state, "answer": result.content, "context": []}


# ── Node 4: Clarify ───────────────────────────────────────────────────────────

def clarify_node(state: AgentState) -> AgentState:
    return {
        **state,
        "answer":  "Your question is a bit unclear. Could you rephrase it or add more detail?",
        "context": []
    }


# ── Routing edge ──────────────────────────────────────────────────────────────

def route_decision(state: AgentState) -> Literal["rag", "general", "clarify"]:
    return state["route"]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("router",  router_node)
    graph.add_node("rag",     rag_node)
    graph.add_node("general", general_node)
    graph.add_node("clarify", clarify_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_decision,
        {
            "rag":     "rag",
            "general": "general",
            "clarify": "clarify"
        }
    )

    graph.add_edge("rag",     END)
    graph.add_edge("general", END)
    graph.add_edge("clarify", END)

    return graph.compile()


# ── Main entry point ──────────────────────────────────────────────────────────

def run_agent(question: str, memory: list, retriever) -> dict:
    graph = build_graph()

    initial_state = AgentState(
        question = question,
        route    = "",
        context  = [],
        answer   = "",
        memory   = memory
    )

    result = graph.invoke(initial_state)

    return {
        "answer":  result["answer"],
        "route":   result["route"],
        "context": result["context"]
    }