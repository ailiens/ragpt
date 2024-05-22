from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import dotenv
import streamlit as st
import time
import os

dotenv.load_dotenv()

# 현재 작업 디렉토리 확인
current_directory = os.getcwd()
st.write(f"현재 작업 디렉토리: {current_directory}")

# Streamlit UI
st.title("👩‍🚀RAGpt👩‍🚀")

# 사이드바에 업로드 창
st.sidebar.header("파일 또는 URL 사용하기")
url_input = st.sidebar.text_input("URL 주소 입력:")
uploaded_files = st.sidebar.file_uploader("파일을 업로드하세요.", type=["pdf", "docx", "txt"], accept_multiple_files=True)
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.0)
top_k = st.sidebar.slider('Top K', min_value=1, max_value=10, value=3)  # 기본값 조정
score_threshold = st.sidebar.slider('Score Threshold', min_value=0.0, max_value=1.0, value=0.5)  # 기본값 조정

# 데이터 로드
all_data = []


def load_file(uploaded_file):
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_file_path = os.path.join(current_directory, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if file_extension == ".pdf":
        loader = PyPDFLoader(temp_file_path)
    elif file_extension == ".docx":
        loader = UnstructuredWordDocumentLoader(temp_file_path)
    elif file_extension == ".txt":
        loader = TextLoader(temp_file_path)
    return loader.load()


if uploaded_files:
    for uploaded_file in uploaded_files:
        start_time = time.time()
        data = load_file(uploaded_file)
        for doc in data:
            doc.metadata['source'] = uploaded_file.name
        load_time = time.time() - start_time
        st.sidebar.write(f"{uploaded_file.name} 파일 로드 시간: {load_time:.2f} 초")
        all_data.extend(data)

    # 메타데이터 확인
    st.sidebar.header("메타데이터 확인")
    for i, doc in enumerate(all_data):
        st.sidebar.write(f"문서 {i + 1} 메타데이터:", doc.metadata)

elif url_input:
    start_time = time.time()
    loader = WebBaseLoader(url_input)
    data = loader.load()
    load_time = time.time() - start_time
    st.sidebar.write(f"URL 로드 시간: {load_time:.2f} 초")
    all_data.extend(data)

# ChromaDB 저장 경로
CHROMA_DB_PATH = "rag_db"

# 데이터가 있는 경우 처리
if all_data:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)  # 분할 전략 개선
    all_splits = text_splitter.split_documents(all_data)

    # Chroma 인스턴스 생성 시 경로 지정
    vectorstore = Chroma.from_documents(
        documents=all_splits,
        embedding=OpenAIEmbeddings(),
        persist_directory=CHROMA_DB_PATH  # 데이터베이스 파일의 경로 지정
    )
    retriever = vectorstore.as_retriever(search_type="similarity_score_threshold",
                                         search_kwargs={"score_threshold": score_threshold, 'k': top_k})

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
            full_response = result['output']
            intermediate_steps = result.get('intermediate_steps', [])

            # 출처 정보 추가
            sources = []
            for step in intermediate_steps:
                if 'document' in step and 'metadata' in step['document']:
                    metadata = step['document']['metadata']
                    if 'source' in metadata and 'page' in metadata:
                        sources.append(f"{metadata['source']} (page {metadata['page']})")
                    elif 'source' in metadata:
                        sources.append(f"{metadata['source']}")

            source_info = "\n\n출처:\n" + "\n".join(sources) if sources else "\n\n출처 정보를 찾을 수 없습니다."
            full_response += source_info

            message_placeholder.markdown(full_response)
        else:
            full_response = "Please upload file or valid URL for search information."
            message_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
