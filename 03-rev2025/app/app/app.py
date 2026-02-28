import os
import json
import random
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from db.db import init_db, save_feedback, save_qa_exchange
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.utils import Secret
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


TEMPLATE = """
You're a FAQ database assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION. CONTEXT may included markdown. Return as markdown format. If the answers is not in CONTEXT, return "Answer is not available in the FAQ database."
Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""


load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QUESTIONS_JSON_PATH = Path(__file__).resolve().parent / "doc_with_q_4o-mini.json"
UNAVAILABLE_ANSWER = "Answer is not available in the FAQ database."
EXAMPLE_SUFFIX = " please use own worrds"


def build_rag_pipeline() -> Pipeline:
    document_store = QdrantDocumentStore(
        url=QDRANT_URL,
        index="hybrid",
        recreate_index=False,
        embedding_dim=512,
        return_embedding=True,
        use_sparse_embeddings=True,
        sparse_idf=True,
    )
    prompt_builder = PromptBuilder(template=TEMPLATE, required_variables=[
                                   "question", "documents"])
    generator = GoogleGenAIChatGenerator(
        model="gemini-2.5-flash",
        api_key=Secret.from_token(GOOGLE_API_KEY),
    )

    rag = Pipeline()
    rag.add_component("sparse_text_embedder",
                      FastembedSparseTextEmbedder(model="Qdrant/bm25"))
    rag.add_component(
        "dense_text_embedder",
        FastembedTextEmbedder(model="jinaai/jina-embeddings-v2-small-en"),
    )
    rag.add_component("retriever", QdrantHybridRetriever(
        document_store=document_store, top_k=5))
    rag.add_component("ranker", FastembedRanker(top_k=5))
    rag.add_component("prompt_builder", prompt_builder)
    rag.add_component("llm", generator)
    rag.add_component("answer_builder", AnswerBuilder())

    rag.connect("sparse_text_embedder.sparse_embedding",
                "retriever.query_sparse_embedding")
    rag.connect("dense_text_embedder.embedding", "retriever.query_embedding")
    rag.connect("retriever.documents", "ranker.documents")
    rag.connect("ranker.documents", "prompt_builder.documents")
    rag.connect("prompt_builder", "llm")
    rag.connect("llm.replies", "answer_builder.replies")
    rag.connect("ranker.documents", "answer_builder.documents")
    return rag


@st.cache_resource
def get_pipeline() -> Pipeline:
    return build_rag_pipeline()


def get_answer(query: str) -> tuple[str, str, str]:
    if not query.strip():
        return "Please enter a question.", "", ""
    if not QDRANT_URL:
        return "Missing QDRANT_URL in environment.", "", ""
    if not GOOGLE_API_KEY:
        return "Missing GOOGLE_API_KEY in environment.", "", ""

    response = get_pipeline().run(
        {
            "sparse_text_embedder": {"text": query},
            "dense_text_embedder": {"text": query},
            "ranker": {"query": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
    )

    answers = response.get("answer_builder", {}).get("answers", [])
    if not answers:
        return "No answer generated.", "", ""

    answer = answers[0]
    answer_text = answer.data if hasattr(answer, "data") else str(answer)

    linked_document = ""
    linked_source = ""
    answer_documents = getattr(answer, "documents", None) or []
    if answer_documents:
        first_document = answer_documents[0]
        meta = getattr(first_document, "meta", {}) or {}
        linked_document = meta.get("content", "") or getattr(first_document, "content", "")
        linked_source = meta.get("source", "")

    if not linked_document:
        ranked_documents = response.get("ranker", {}).get("documents", [])
        if ranked_documents:
            first_document = ranked_documents[0]
            meta = getattr(first_document, "meta", {}) or {}
            linked_document = meta.get("content", "") or getattr(first_document, "content", "")
            linked_source = meta.get("source", "")

    return answer_text, linked_document, linked_source


@st.cache_data
def load_example_questions() -> list[str]:
    if not QUESTIONS_JSON_PATH.exists():
        return []

    questions: list[str] = []
    try:
        with QUESTIONS_JSON_PATH.open(mode="r", encoding="utf-8") as f:
            rows = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

    if not isinstance(rows, list):
        return []

    for row in rows:
        if not isinstance(row, dict):
            continue
        question = (row.get("question") or "").strip()
        if question:
            questions.append(question)
    return questions


def pick_random_example_question() -> str:
    questions = load_example_questions()
    if not questions:
        return ""
    return random.choice(questions)


def main() -> None:
    st.set_page_config(page_title="FAQ RAG", page_icon=":mag:")
    st.title("FAQ RAG")
    init_db()

    if "awaiting_feedback" not in st.session_state:
        st.session_state["awaiting_feedback"] = False
    if "last_query" not in st.session_state:
        st.session_state["last_query"] = ""
    if "last_answer" not in st.session_state:
        st.session_state["last_answer"] = ""
    if "last_feedback" not in st.session_state:
        st.session_state["last_feedback"] = ""
    if "last_linked_document" not in st.session_state:
        st.session_state["last_linked_document"] = ""
    if "last_linked_source" not in st.session_state:
        st.session_state["last_linked_source"] = ""
    if "last_exchange_id" not in st.session_state:
        st.session_state["last_exchange_id"] = None
    if "last_db_warning" not in st.session_state:
        st.session_state["last_db_warning"] = ""

    if "example_question" not in st.session_state:
        st.session_state["example_question"] = pick_random_example_question()

    example_question = st.session_state.get("example_question", "")
    example_placeholder = (
        f"{example_question}{EXAMPLE_SUFFIX}" if example_question else ""
    )

    if st.button("New example question"):
        st.session_state["example_question"] = pick_random_example_question()
        st.rerun()

    awaiting_feedback = st.session_state["awaiting_feedback"]
    query = st.text_input(
        "Ask a question",
        placeholder=example_placeholder,
        key="query_input",
        disabled=awaiting_feedback,
    )

    if st.button("Get answer", type="primary", disabled=awaiting_feedback):
        with st.spinner("Searching and generating answer..."):
            st.session_state["last_db_warning"] = ""
            answer, linked_document, linked_source = get_answer(query)
            exchange_id = None
            if query.strip():
                exchange_id = save_qa_exchange(query=query, answer=answer)
                if exchange_id is None:
                    st.session_state["last_db_warning"] = "Database save failed: question/answer was not persisted."
            st.session_state["last_query"] = query
            st.session_state["last_answer"] = answer
            st.session_state["last_linked_document"] = linked_document
            st.session_state["last_linked_source"] = linked_source
            st.session_state["last_exchange_id"] = exchange_id
            st.session_state["awaiting_feedback"] = True
            st.session_state["last_feedback"] = ""
            st.rerun()

    if st.session_state["last_answer"]:
        st.markdown(st.session_state["last_answer"])

    if st.session_state["awaiting_feedback"]:
        st.info("Please provide feedback on the answer before asking a new question.")
        col_up, col_down = st.columns(2)
        if col_up.button("👍 Helpful", use_container_width=True):
            st.session_state["last_feedback"] = "up"
            if st.session_state["last_exchange_id"] is not None:
                ok = save_feedback(st.session_state["last_exchange_id"], "up")
                if not ok:
                    st.session_state["last_db_warning"] = "Database save failed: feedback was not persisted."
            else:
                st.session_state["last_db_warning"] = "Feedback was not saved because the related answer was not persisted."
            st.session_state["awaiting_feedback"] = False
            st.rerun()
        if (
            st.session_state["last_answer"].strip() != UNAVAILABLE_ANSWER
            and st.session_state["last_linked_document"].strip()
        ):
            col_up.markdown("**Ranked First Document:**")
            col_up.markdown(st.session_state["last_linked_document"])
            if st.session_state["last_linked_source"].strip():
                col_up.markdown(f"**Linked Source:** {st.session_state['last_linked_source']}")
        if col_down.button("👎 Not helpful", use_container_width=True):
            st.session_state["last_feedback"] = "down"
            if st.session_state["last_exchange_id"] is not None:
                ok = save_feedback(st.session_state["last_exchange_id"], "down")
                if not ok:
                    st.session_state["last_db_warning"] = "Database save failed: feedback was not persisted."
            else:
                st.session_state["last_db_warning"] = "Feedback was not saved because the related answer was not persisted."
            st.session_state["awaiting_feedback"] = False
            st.rerun()
    elif st.session_state["last_feedback"] == "up":
        st.success("Thanks for the feedback: 👍")
    elif st.session_state["last_feedback"] == "down":
        st.success("Thanks for the feedback: 👎")

    if st.session_state["last_db_warning"]:
        st.warning(st.session_state["last_db_warning"])


if __name__ == "__main__":
    main()
