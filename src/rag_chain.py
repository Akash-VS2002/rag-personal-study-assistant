"""
RAG chain: wires a Chroma retriever to a Groq-hosted LLM with a
study-assistant-focused prompt, and returns the answer plus sources.

Includes conversational memory: follow-up questions are first
rewritten into standalone questions using chat history, so retrieval
still works for things like "explain that in more detail" or
"what about the second one".
"""

from typing import Dict, List, Tuple

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever

from src import config

# --- Prompt used to rewrite a follow-up question into a standalone one ---
CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question,
which might reference context in the chat history, rewrite it as a standalone
question that can be understood without the chat history. Do NOT answer the
question — only rewrite it if needed, otherwise return it as-is."""

# --- Prompt used to actually answer, given retrieved context ---
QA_SYSTEM_PROMPT = """You are a helpful, precise study assistant.
Answer the user's question using ONLY the provided context from their notes/textbook.
If the answer is not present in the context, say you don't have that information in the
uploaded material instead of guessing. Keep answers clear and exam-oriented: define terms,
give short explanations, and use bullet points where useful.

Context:
{context}
"""


def get_llm() -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Add it to your .env file (see .env.example)."
        )
    return ChatGroq(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_MODEL_NAME,
        temperature=config.LLM_TEMPERATURE,
    )


def to_langchain_messages(chat_history: List[Tuple[str, str]]) -> List[BaseMessage]:
    """Convert a list of (role, content) tuples into LangChain message objects."""
    messages: List[BaseMessage] = []
    for role, content in chat_history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def build_conversational_rag_chain(vectorstore: Chroma):
    """
    Build a history-aware RAG chain:
      1. history_aware_retriever: rewrites the question using chat history,
         then retrieves relevant chunks for the rewritten question.
      2. question_answer_chain: stuffs retrieved chunks into the QA prompt
         and generates the final answer.
    """
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain


def answer_question(
    vectorstore: Chroma,
    question: str,
    chat_history: List[Tuple[str, str]] = None,
) -> Dict:
    """
    Run the conversational RAG chain and return the answer plus cited sources.

    chat_history: list of (role, content) tuples, e.g.
        [("user", "What is backpropagation?"), ("assistant", "...")]
    Pass the history BEFORE the current question — don't include `question` in it.
    """
    chat_history = chat_history or []
    rag_chain = build_conversational_rag_chain(vectorstore)

    result = rag_chain.invoke(
        {
            "input": question,
            "chat_history": to_langchain_messages(chat_history),
        }
    )

    sources = result.get("context", [])
    return {
        "answer": result["answer"],
        "sources": [
            {
                "file": d.metadata.get("source_file", "unknown"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:200],
            }
            for d in sources
        ],
    }