import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    ReadTheDocsLoader,
)  # helps building documentation for Github repositories
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
)  # helps splitting text into sentences and words
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
from consts import INDEX_NAME #pinecone index name is declared in consts.py

load_dotenv()
pc = Pinecone(api_key="PINECONE_API_KEY")


def ingest_docs() -> None:

    loader = ReadTheDocsLoader(
        path=r"langchain-docs\langchain.readthedocs.io\en\latest", encoding="ISO-8859-1"
    )
    # loader = ReadTheDocsLoader(path= "langchain-docs/api.python.langchain.com/en/latest/adapters", encoding="UTF-8")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} documents")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_documents)
    print(f"Split into {len(documents)} chunks")

    for doc in documents:
        old_path = doc.metadata["source"]
        new_url = old_path.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Inserting {len(documents)} documents to Pinecone....")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    PineconeStore.from_documents(
        documents, embeddings, index_name=INDEX_NAME
    )  # storing vectors and chunks in the pinecone serverless index (cosine)
    print("******Vectors added to Pinecone!******")


if __name__ == "__main__":
    ingest_docs()
