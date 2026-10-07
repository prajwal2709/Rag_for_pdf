"""Streamlit interface for asking questions about uploaded documents."""

from __future__ import annotations
import os
import hashlib
import io
import uuid

import streamlit as st
from deepagents import create_deep_agent
from deepagents.backends import StateBackend
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pymupdf 
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()


from prompts import (
    CHUNK_ANALYST_INSTRUCTIONS,
    RAG_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)


def apply_dashboard_theme() -> None:
    """Apply the dark emerald dashboard styling used by the project thumbnail."""
    st.markdown(
        """
        <style>
        :root {
            --emerald: #18b66a;
            --emerald-bright: #4ade80;
            --emerald-dark: #087443;
            --surface: #111716;
            --surface-raised: #17201e;
            --border: #2a3834;
            --muted: #9aaba5;
        }

        .stApp {
            color: #f4f7f5;
            background:
                radial-gradient(circle at 85% 0%, rgba(18, 143, 83, 0.20), transparent 32rem),
                radial-gradient(circle at 10% 100%, rgba(5, 95, 54, 0.16), transparent 28rem),
                #090d0c;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101715 0%, #0b100f 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label {
            color: #c9d4d0;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 4rem;
        }

        .rag-hero {
            position: relative;
            overflow: hidden;
            margin-bottom: 1.5rem;
            padding: 1.7rem 2rem;
            border: 1px solid #254237;
            border-radius: 18px;
            background: linear-gradient(115deg, rgba(21, 44, 36, .96), rgba(14, 21, 19, .96));
            box-shadow: 0 18px 50px rgba(0, 0, 0, .28);
        }

        .rag-hero::after {
            content: "";
            position: absolute;
            width: 280px;
            height: 280px;
            right: -80px;
            top: -150px;
            border-radius: 50%;
            background: rgba(24, 182, 106, .18);
            filter: blur(20px);
        }

        .rag-eyebrow {
            margin-bottom: .45rem;
            color: var(--emerald-bright);
            font-size: .76rem;
            font-weight: 800;
            letter-spacing: .16em;
            text-transform: uppercase;
        }

        .rag-title {
            margin: 0;
            color: #ffffff;
            font-size: clamp(2rem, 4vw, 3.25rem);
            font-weight: 850;
            letter-spacing: -.045em;
            line-height: 1;
        }

        .rag-title span { color: var(--emerald-bright); }

        .rag-subtitle {
            max-width: 690px;
            margin: .85rem 0 0;
            color: var(--muted);
            font-size: 1rem;
        }

        [data-testid="stFileUploaderDropzone"],
        [data-testid="stChatMessage"],
        div[data-testid="stAlert"] {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: rgba(20, 29, 27, .90);
        }

        [data-testid="stChatMessage"] {
            margin-bottom: .7rem;
            padding: .35rem .45rem;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
            border-left: 3px solid var(--emerald);
        }

        .stButton > button[kind="primary"] {
            border: 1px solid #31ce80;
            background: linear-gradient(135deg, #15975a, #087443);
            color: white;
            font-weight: 750;
            box-shadow: 0 8px 24px rgba(18, 166, 93, .22);
        }

        .stButton > button[kind="primary"]:hover {
            border-color: #65e59d;
            background: linear-gradient(135deg, #1db86c, #09864d);
        }

        .stButton > button[kind="secondary"] {
            border-color: var(--border);
            background: var(--surface-raised);
            color: #dbe5e1;
        }

        [data-testid="stChatInput"] {
            border: 1px solid #315144;
            border-radius: 14px;
            background: #121a18;
            box-shadow: 0 10px 35px rgba(0, 0, 0, .25);
        }

        [data-testid="stChatInput"]:focus-within {
            border-color: var(--emerald);
            box-shadow: 0 0 0 2px rgba(24, 182, 106, .14);
        }

        [data-testid="stFileUploaderDropzone"] button {
            border-color: #277e55;
            background: #163326;
            color: #dff8e9;
        }

        h1, h2, h3 { letter-spacing: -.025em; }
        a { color: var(--emerald-bright) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def uploaded_file_to_document(uploaded_file) -> Document:
    """Extract text from a supported Streamlit UploadedFile."""
    raw_data = uploaded_file.getvalue()
    suffix = uploaded_file.name.rsplit(".", 1)[-1].lower()

    if suffix == "pdf":
        pdf = pymupdf.open(stream=raw_data, filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
        pdf.close()
    else:
        text = raw_data.decode("utf-8", errors="replace")

    if not text.strip():
        raise ValueError(f"No readable text was found in {uploaded_file.name}.")

    return Document(page_content=text, metadata={"source": uploaded_file.name})


def file_set_id(uploaded_files) -> str:
    """Return a stable identifier for the currently uploaded file contents."""
    digest = hashlib.sha256()
    for uploaded_file in uploaded_files:
        digest.update(uploaded_file.name.encode("utf-8"))
        digest.update(uploaded_file.getvalue())
    return digest.hexdigest()

def build_agent(documents: list[Document]):
    """Build a session-local vector index and Deep Agent."""

    model = ChatOpenAI(
        model="openai/gpt-5.5",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        max_tokens=500,
    )

    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-large",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    vector_store = InMemoryVectorStore(embeddings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = splitter.split_documents(documents)
    vector_store.add_documents(chunks)

    backend = StateBackend()

    @tool(parse_docstring=True)
    def search_documentation(query: str) -> str:
        """Search uploaded documents and save matching chunks to the agent filesystem.

        Args:
            query: Natural-language search query.

        Returns:
            File paths where retrieved chunks were saved under /retrieved/.
        """

        retrieved_docs = vector_store.similarity_search(
            query,
            k=min(4, len(chunks)),
        )

        batch_id = uuid.uuid4().hex[:8]

        uploads: list[tuple[str, bytes]] = []
        saved_paths: list[str] = []

        for index, doc in enumerate(retrieved_docs, start=1):
            path = f"/retrieved/{batch_id}/chunk_{index}.md"

            content = (
                f"# Source: {doc.metadata.get('source', 'unknown')}\n\n"
                f"{doc.page_content}"
            )

            uploads.append(
                (path, content.encode("utf-8"))
            )

            saved_paths.append(path)

        backend.upload_files(uploads)

        return (
            f"Saved {len(saved_paths)} document chunks:\n"
            + "\n".join(saved_paths)
        )

    instructions = (
        RAG_WORKFLOW_INSTRUCTIONS
        + "\n\n"
        + SUBAGENT_DELEGATION_INSTRUCTIONS.format(
            max_concurrent_analysts=3
        )
    )

    analyst = {
        "name": "chunk-analyst",
        "description": (
            "Analyze one retrieved document chunk for the user's question. "
            "Read the assigned file using read_file and return the relevant facts."
        ),
        "model": model,
        "tools": [],
        "system_prompt": CHUNK_ANALYST_INSTRUCTIONS,
    }

    agent = create_deep_agent(
        model=model,
        tools=[search_documentation],
        backend=backend,
        system_prompt=instructions,
        subagents=[analyst],
    )

    return agent, len(chunks)

def final_text(result: dict) -> str:
    """Extract the last readable model message from an agent result."""

    for message in reversed(result.get("messages", [])):
        content = getattr(message, "content", "")

        if isinstance(content, str) and content.strip():
            return content

        if isinstance(content, list):
            text_parts = []

            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

            if text_parts:
                return "\n".join(text_parts)

    return "The agent completed the request but did not return a text response."

def main() -> None:
    st.set_page_config(page_title="Document Q&A", page_icon="📚", layout="wide")
    apply_dashboard_theme()
    st.markdown(
        """
        <section class="rag-hero">
            <div class="rag-eyebrow">LangChain · Retrieval augmented generation</div>
            <h1 class="rag-title">Chat with your <span>knowledge</span></h1>
            <p class="rag-subtitle">
                Upload your documents, build a semantic index, and get grounded answers
                from an agent that searches before it speaks.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("### ◈ Knowledge base")

        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["pdf", "txt", "md"],
            accept_multiple_files=False,
        )

        process_clicked = st.button(
            "Process document",
            type="primary",
            disabled=uploaded_file is None,
            use_container_width=True,
        )

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if process_clicked and uploaded_file is not None:
        try:
            with st.spinner("Reading, splitting, and indexing document..."):
                document = uploaded_file_to_document(uploaded_file)
                agent, chunk_count = build_agent([document])

            st.session_state.agent = agent
            st.session_state.file_set_id = file_set_id([uploaded_file])
            st.session_state.messages = []
            st.success(f"Indexed {uploaded_file.name} into {chunk_count} chunks.")
        except Exception as exc:
            st.error(f"Could not process the document: {exc}")

    current_file_set_id = file_set_id([uploaded_file]) if uploaded_file else None
    document_is_current = (
        "agent" in st.session_state
        and current_file_set_id is not None
        and current_file_set_id == st.session_state.get("file_set_id")
    )

    if uploaded_file and "agent" in st.session_state and not document_is_current:
        st.warning("The selected file changed. Process it before asking questions.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "Ask a question about your documents",
        disabled=not document_is_current,
    )
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching and analyzing your documents..."):
                try:
                    result = st.session_state.agent.invoke(
                        {"messages": [HumanMessage(content=question)]}
                    )
                    answer = final_text(result)
                except Exception as exc:
                    answer = f"I couldn't answer that question: {exc}"
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    if "agent" not in st.session_state:
        st.info("Upload and process at least one document to begin.")


if __name__ == "__main__":
    main()
