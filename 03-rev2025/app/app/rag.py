import os
from dotenv import load_dotenv
from haystack import Pipeline
from haystack.components.builders import PromptBuilder
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack_integrations.components.embedders.fastembed import (
    FastembedSparseTextEmbedder,
    FastembedTextEmbedder,
)
from haystack_integrations.components.rankers.fastembed import FastembedRanker
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator



TEMPLATE = """
You're a FAQ database assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""


URL = "http://localhost:6333"




def DocumentStore(url:str):
    document_store = QdrantDocumentStore(
        url=URL,
        index="hybrid",
        recreate_index=True,
        embedding_dim=512,
        return_embedding=True,
        use_sparse_embeddings=True,
        sparse_idf=True,
    )
    return document_store


def PromptBuilder(template:str=TEMPLATE):
    prompt_builder = PromptBuilder(
        template=TEMPLATE, required_variables=["question", "documents"]
    )
    return prompt_builder


def Generator(api_key: str):
    generator = GoogleGenAIChatGenerator(model="gemini-2.5-flash", api_key=api_key)
    return generator


def HybridRAGPipeline():

    rag = Pipeline()
    rag.add_component("sparse_text_embedder",FastembedSparseTextEmbedder(model="Qdrant/bm25"))
    rag.add_component("dense_text_embedder", FastembedTextEmbedder(model="jinaai/jina-embeddings-v2-small-en"))
    rag.add_component("retriever", QdrantHybridRetriever(document_store=document_store, top_k=5))
    rag.add_component("ranker", FastembedRanker(top_k=5))
    rag.add_component("prompt_builder", prompt_builder)
    rag.add_component("llm", generator)
    rag.add_component(instance=AnswerBuilder(), name="answer_builder")
    rag.connect("sparse_text_embedder.sparse_embedding","retriever.query_sparse_embedding")
    rag.connect("dense_text_embedder.embedding","retriever.query_embedding")
    rag.connect("retriever.documents", "ranker.documents")
    rag.connect("ranker.documents", "prompt_builder")
    rag.connect("prompt_builder", "llm")
    rag.connect("llm.replies", "answer_builder.replies")
    rag.connect("retriever", "answer_builder.documents")

    return rag


def GetAnswer(query):
    document_store = DocumentStore(url=URL)
    prompt_builder = PromptBuilder(template=TEMPLATE)
    generator = Generator(api_key=os.getenv("GOOGLE_API_KEY"))
    rag = HybridRAGPipeline()
    response = rag.run(
        {
            "sparse_text_embedder": {"text": query},
            "dense_text_embedder": {"text": query},
            "ranker": {"query": query},
            "prompt_builder": {"question": query},
            "answer_builder": {"query": query},
        }
    )

