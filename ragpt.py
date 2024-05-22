from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import dotenv
import streamlit as st
import time
import os

dotenv.load_dotenv()

# Streamlit UI
st.title("👩‍🚀RAGpt👩‍🚀")

# 사이드바에 업로드 창
st.sidebar.header("파일 또는 URL 사용하기")
url_input = st.sidebar.text_input("URL 주소 입력:")
uploaded_files = st.sidebar.file_uploader("파일을 업로드하세요.", type=["pdf", "docx", "txt"], accept_multiple_files=True)
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0)
top_k = st.sidebar.slider('Top K', min_value=1, max_value=10, value=1)
# 데이터 로드
all_data = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1]
        start_time = time.time()
        if file_extension == ".pdf":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = PyPDFLoader(uploaded_file.name)
            data = loader.load()
        elif file_extension == ".docx":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = UnstructuredWordDocumentLoader(uploaded_file.name)
            data = loader.load()
        elif file_extension == ".txt":
            with open(uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())
            loader = TextLoader(uploaded_file.name)
            data = loader.load()

        load_time = time.time() - start_time
        st.sidebar.write(f"{uploaded_file.name} 파일 로드 시간: {load_time:.2f} 초")
        all_data.extend(data)

elif url_input:
    start_time = time.time()
    loader = WebBaseLoader(url_input)
    data = loader.load()
    load_time = time.time() - start_time
    st.sidebar.write(f"URL 로드 시간: {load_time:.2f} 초")
    all_data.extend(data)

# 데이터가 있는 경우 처리
if all_data:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    all_splits = text_splitter.split_documents(all_data)

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"score_threshold": 0.6, 'k': top_k})

    from langchain.agents.agent_toolkits import create_retriever_tool

    tool = create_retriever_tool(
        retriever,
        "pdf_search",
        "가능한 모든 정보는 PDF 문서 내에서 검색합니다.",
    )
    tools = [tool]

    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(temperature=temperature)

    # This is needed for both the memory and the prompt
    memory_key = "history"

    from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
        AgentTokenBufferMemory,
    )

    memory = AgentTokenBufferMemory(memory_key=memory_key, llm=llm)

    from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
    from langchain.schema.messages import SystemMessage
    from langchain.prompts import MessagesPlaceholder

    system_message = SystemMessage(
        content=(
            "You are a nice customer service agent."
            "Do your best to answer the questions. "
            "Feel free to use any tools available to look up "
            "relevant information, only if necessary"
            "If you don't know the answer, just say you don't know. Don't try to make up an answer."
            "Make sure to answer in Korean"
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=system_message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name=memory_key)],
    )

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)

    from langchain.agents import AgentExecutor

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        return_intermediate_steps=True,
    )

# 채팅 UI
st.header("Upload or URL & ask information U need")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me in the document"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 데이터를 사용하여 에이전트 실행
        if all_data:
            result = agent_executor({"input": prompt})
            for chunk in result['output'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        else:
            full_response = "Please upload file or valid URL for search information."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
