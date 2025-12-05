import json
import unicodedata
import streamlit as st
from dotenv import load_dotenv
from rapidfuzz import process, fuzz

# LangChain (LLM + VectorStore)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 🔥 고급 retrievers

from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor  # compression
from langchain_community.retrievers.contextual_compression import (
    ContextualCompressionRetriever
)

# LCEL + Message History
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory


# ============================================================
# 1. UI / APP 초기화
# ============================================================

def init_page():
    st.set_page_config(page_title="영어 학습 도우미", page_icon="📚")
    st.title("📚 영어 학습 도우미")
    st.markdown("영상 기반 맞춤형 학습 시스템")


def init_env():
    load_dotenv()


AVAILABLE_TOPICS = """
현재 학습 가능한 주제:
1. 시간 전치사 (at, on, in)
2. 장소 전치사 (at, on, in)
3. 소유격 (my, mine, his)
4. Does he? vs Is he?
5. 수동태 (be + 과거분사)
6. 현재완료
7. that 용법
8. I'm not used to 패턴
9. Do you? vs Are you?
"""


# ============================================================
# 2. 데이터 로딩
# ============================================================

@st.cache_resource
def load_vectorstore():
    with st.spinner("벡터스토어 로딩 중..."):
        embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="english_grammar_chunked",
        )
    return vectorstore


@st.cache_resource
def load_knowledge_graph():
    with open(
        "/data/edutem/sooine/rag_bot/knowledge_graph.json",
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


# ============================================================
# 3. 지식 그래프 검색 유틸
# ============================================================

def normalize(s: str):
    return unicodedata.normalize("NFC", s.lower().replace(" ", ""))


def fuzzy_match_topic(query, topic_list):
    q = normalize(query)
    candidates = [normalize(t) for t in topic_list]
    match, score, idx = process.extractOne(q, candidates, scorer=fuzz.ratio)
    return topic_list[idx] if score > 70 else None


def search_in_knowledge_graph(query, knowledge_graph):
    q = query.lower().strip()
    topic_list = list(knowledge_graph.keys())

    best = fuzzy_match_topic(query, topic_list)
    if best:
        return {"type": "main_topic", "main_topic": best, "data": knowledge_graph[best]}

    # 정확 일치
    for topic in topic_list:
        if topic.lower() == q:
            return {"type": "main_topic", "main_topic": topic, "data": knowledge_graph[topic]}

    # 부분 일치
    for topic in topic_list:
        if q in topic.lower():
            return {"type": "main_topic", "main_topic": topic, "data": knowledge_graph[topic]}

    # Sub-topic 검색
    best_match = None
    best_score = 0

    for main_topic, topic_data in knowledge_graph.items():
        for sub_id, sub_data in topic_data["sub_topics"].items():
            score = 0
            if q in sub_data["title"].lower(): score += 3
            if q in sub_data["concept"].lower(): score += 2
            for ex in sub_data.get("examples", []):
                if q in ex.lower():
                    score += 1
                    break
            if score > best_score:
                best_score = score
                best_match = {
                    "type": "sub_topic",
                    "main_topic": main_topic,
                    "sub_topic_id": sub_id,
                    "data": sub_data,
                }

    if best_match and best_score >= 1:
        return best_match

    return None


# ============================================================
# 4. 세션 & 히스토리
# ============================================================

def init_session_state():
    if "store" not in st.session_state:
        st.session_state["store"] = {}
    if "session_id" not in st.session_state:
        st.session_state["session_id"] = "default_user"
    if "mode" not in st.session_state:
        st.session_state["mode"] = "search"


def get_session_history(session_id):
    store = st.session_state["store"]
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ============================================================
# 5. LLM / Retriever / RAG 구성
# ============================================================

def init_llm():
    return ChatOpenAI(model="gpt-4o", temperature=0)


def build_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 영어 학습 도우미입니다. 아래 context를 바탕으로 답변하세요.\n\n<context>\n{context}\n</context>"),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}")
    ])


# 🔥 핵심 — MultiQuery → Ensemble → Compression

def build_advanced_retriever(vectorstore, llm_for_retrieval):
    # 1) Base retriever (벡터 검색)
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

    # 2) LLM 기반 문서 압축기
    compressor = LLMChainExtractor.from_llm(llm_for_retrieval)

    # 3) 압축 retriever = base_retriever + compressor
    compression_retriever = ContextualCompressionRetriever(
        base_retriever=base_retriever,
        base_compressor=compressor
    )

    return compression_retriever


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(retriever, llm, prompt):
    def rag_with_chain(inputs):
        q = inputs["question"]
        hist = inputs.get("chat_history", [])

        docs = retriever.invoke(q)

        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": format_docs(docs),
            "question": q,
            "chat_history": hist,
        })

        return {"answer": answer, "source_docs": docs}

    rag_chain = RunnableLambda(rag_with_chain)

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history"
    )


# ============================================================
# 6. UI
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 설정")
        mode = st.radio("모드 선택", ["🔍 Search", "📝 Quiz", "📖 Review"], index=0)
        st.session_state["mode"] = mode
        st.divider()
        st.markdown("### 📚 학습 주제")
        st.markdown(AVAILABLE_TOPICS)


def render_chat(chain, knowledge_graph):
    session_id = st.session_state["session_id"]
    history = get_session_history(session_id)

    # 기존 대화 출력
    for msg in reversed(history.messages):
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    user_input = st.chat_input("질문을 입력하세요...")
    if not user_input:
        return

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------------
    # 🔥 1) 지식 그래프 우선 검색
    # -------------------------------
    kg_result = search_in_knowledge_graph(user_input, knowledge_graph)

    if kg_result:
        with st.chat_message("assistant"):
            st.markdown("### 📘 지식 그래프 설명")

            if kg_result["type"] == "main_topic":
                st.write(kg_result["data"]["concept"])
                st.markdown("#### 예문")
                for ex in kg_result["data"].get("examples", []):
                    st.write(f"- {ex}")
            else:
                st.write(kg_result["data"]["concept"])
                st.markdown("#### 예문")
                for ex in kg_result["data"].get("examples", []):
                    st.write(f"- {ex}")
        return

    # -------------------------------
    # 🔥 2) 지식 그래프에 없으면 → RAG
    # -------------------------------
    with st.chat_message("assistant"):
        result = chain.invoke(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        st.markdown("### 💡 설명")
        st.write(result["answer"])



# ============================================================
# 7. main()
# ============================================================

def main():
    init_page()
    init_env()
    init_session_state()

    vectorstore = load_vectorstore()
    knowledge_graph = load_knowledge_graph()

    llm = init_llm()
    prompt = build_prompt()
    retriever = build_advanced_retriever(vectorstore, llm)
    chain = build_rag_chain(retriever, llm, prompt)

    render_sidebar()
    render_chat(chain, knowledge_graph)


if __name__ == "__main__":
    main()
