"""
LangGraph version of the RAG pipeline, with persistent memory.

Instead of a linear LangChain chain, this is a real graph with two nodes:

    START -> retrieve -> generate -> END

Conversation history is NOT passed in manually from Streamlit. Instead,
every call is made with a `thread_id` (think: "study session name" /
"notebook name"). LangGraph's SqliteSaver checkpointer automatically:
  - loads that thread's past messages from checkpoints.db before running
  - appends the new question + answer to it
  - saves it back to checkpoints.db after running

This means conversation memory survives app restarts and days between
sessions, as long as you reuse the same thread_id.
"""

import sqlite3
from typing import List, TypedDict

from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import Annotated

from src import config

CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question,
which might reference context in the chat history, rewrite it as a standalone
question that can be understood without the chat history. Do NOT answer the
question — only rewrite it if needed, otherwise return it as-is. Return ONLY
the rewritten question, nothing else."""

QA_SYSTEM_PROMPT = """You are a helpful, precise study assistant.
Answer the user's question using ONLY the provided context from their notes/textbook.
If the answer is not present in the context, say you don't have that information in the
uploaded material instead of guessing. Keep answers clear and exam-oriented: define terms,
give short explanations, and use bullet points where useful.

Context:
{context}
"""


class State(TypedDict):
    """Graph state. `messages` is persisted across turns by the checkpointer."""
    messages: Annotated[List[BaseMessage], add_messages]
    context: List[Document]


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


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source_file', 'unknown')}, "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


def build_graph(vectorstore: Chroma):
    """Compile the LangGraph app with a SQLite checkpointer attached."""
    llm = get_llm()
    retriever = vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    def retrieve(state: State):
        """Node 1: rewrite the question using history (if any), then retrieve chunks."""
        history = state["messages"][:-1]
        latest_question = state["messages"][-1].content

        if history:
            rewritten = llm.invoke(
                contextualize_prompt.format_messages(
                    chat_history=history, input=latest_question
                )
            )
            search_query = rewritten.content
        else:
            search_query = latest_question

        docs = retriever.invoke(search_query)
        return {"context": docs}

    def generate(state: State):
        """Node 2: answer using retrieved context + full conversation history."""
        system = SystemMessage(content=QA_SYSTEM_PROMPT.format(context=format_docs(state["context"])))
        response = llm.invoke([system] + state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)

    # check_same_thread=False: Streamlit can call from different threads
    conn = sqlite3.connect(str(config.CHECKPOINT_DB_PATH), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    return graph_builder.compile(checkpointer=checkpointer)


def ask(graph, question: str, thread_id: str) -> dict:
    """
    Ask a question within a given study session (thread_id).
    LangGraph automatically loads/saves history for this thread_id.
    """
    from langchain_core.messages import HumanMessage

    result = graph.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"configurable": {"thread_id": thread_id}},
    )

    answer = result["messages"][-1].content
    sources = result.get("context", [])

    return {
        "answer": answer,
        "sources": [
            {
                "file": d.metadata.get("source_file", "unknown"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:200],
            }
            for d in sources
        ],
    }


def get_thread_messages(graph, thread_id: str) -> List[BaseMessage]:
    """Load the saved message history for a thread_id, e.g. to render on page load."""
    state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    if state and state.values:
        return state.values.get("messages", [])
    return []