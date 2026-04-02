import streamlit as st  
import os
from langchain_groq import ChatGroq
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_community.utilities import (
    WikipediaAPIWrapper,
    ArxivAPIWrapper,
    DuckDuckGoSearchAPIWrapper,
)
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

# --- Tools ---
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchAPIWrapper()

# --- Streamlit UI ---
st.title("LangChain Chat With Search 🔍")
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API key:", type="password")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant that helps people find information."}
    ]

# display history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

# --- Main Chat ---
if (prompt := st.chat_input("Ask something...")):
    # Store user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Initialize LLM and tools
    llm = ChatGroq(groq_api_key=api_key, model="llama-3.1-8b-instant", streaming=True)
    tools = [search, wiki, arxiv]

    # Create the agent
    search_agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True
    ) 

    # Stream and display assistant reply
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())  # ✅ fixed
        response = search_agent.run(prompt, callbacks=[st_cb])  # ✅ pass prompt
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
