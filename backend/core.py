import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import (
    ConversationalRetrievalChain,
)  # used in place of retrievalqa to integrate memory into chats
from langchain_community.vectorstores import Pinecone as PineconeStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from consts import INDEX_NAME
from typing import Any
from typing import List

load_dotenv()
pc = Pinecone(api_key="PINECONE_API_KEY")


def run_llm(query: str, chat_history: List[tuple[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = PineconeStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0, model_name="gpt-3.5-turbo")
    # qa = RetrievalQA.from_chain_type(
    #     llm=chat,
    #     chain_type="stuff",
    #     retriever=docsearch.as_retriever(),
    #     return_source_documents=True,
    # )  # implementation of retrievalqa - retriever is an object of the vectorstore which matches vectors based on similarity search

    qa = ConversationalRetrievalChain.from_llm(
        llm=chat, retriever=docsearch.as_retriever(), return_source_documents=True
    )  # implementations of the Conversational chain model
    return qa(
        {"question": query, "chat_history": chat_history}
    )  # query key is labelled as question since it throws errors


if __name__ == "__main__":
    print(run_llm("What is LangChain?"))
