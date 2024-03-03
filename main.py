from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
from typing import Set


st.header("LangChain documentation helper bot 🤖")
prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")

# session state captures the event and preserves the state of the app, they share variables for each user session
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []  # what the user asked

if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = (
        []
    )  # what the chatbot answered the formatted response

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = (
        []
    )  # what the chatbot answered the formatted response


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return "No sources found"
    sources_list = list(source_urls)
    sources_list.sort()
    sources_string = "sources: \n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source} \n"
    return sources_string


if prompt:
    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt, chat_history=st.session_state["chat_history"]
        )
        sources = set(
            [doc.metadata["source"] for doc in generated_response["source_documents"]]
        )  # will generate list of urls and is converting it to set to remove duplicates

        formatted_response = f"{generated_response['answer']} \n\n {create_sources_string(sources)}"  # formatted response with sources

        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(
            (prompt, generated_response["answer"])
        )  # passed tuple notation to store history
        # result key throws errors so used answer

if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        message(user_query, is_user=True)  # prints another avatar for the user
        message(generated_response)  # prints another avatar for the chatbot
