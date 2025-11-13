import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit.components.v1 as components
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
import openai
import json
import os

load_dotenv()

# ============================================================
# 1. 초기 설정
# ============================================================

st.set_page_config(
    page_title="영어 학습 도우미",
    page_icon="📚",
)

st.title("📚 영어 학습 도우미")
st.markdown("영상 기반 맞춤형 학습 시스템")

# ============================================================
# 2. 벡터스토어 로드 (캐시)
# ============================================================

@st.cache_resource
def load_vectorstore():    
    with st.spinner("벡터스토어 로딩 중..."):
            embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
        
        # 이미 만들어진 벡터스토어 로드
            vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=embeddings,
            collection_name="lectures"
            )
    
    return vectorstore

vectorstore = load_vectorstore()

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
# 세션 저장소
# ============================================================

if "store" not in st.session_state:
    st.session_state["store"] = {}
    
# 세션 ID
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "default_user"
    
if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "mode" not in st.session_state:
    st.session_state["mode"] = "search"

def get_session_history(session_id):
    if session_id not in st.session_state["store"]:
        st.session_state["store"][session_id] = ChatMessageHistory()
    return st.session_state["store"][session_id]

# ============================================================
# 체인 생성
# ============================================================

prompt = ChatPromptTemplate.from_messages([
    ("system", """당신은 친절한 영어 학습 도우미입니다.

    사용자 질문에 대해 간단명료하게 답변해주고,
    강의에서 설명한 핵심 내용과 예시를 포함해서 3-5문장으로 답변하세요.
    대답할 땐 한국말로 하세요
        """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", """강의 내용:
        {context}

        질문: {question}""")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

# ============================================================
# 3. 세션 상태 초기화
# ============================================================


# ============================================================
# 4. 사이드바
# ============================================================

with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모드 선택
    mode = st.radio(
        "모드 선택",
        ["🔍 Search (검색)", "📝 Quiz (문제)", "📖 Review (복습)"],
        index=0
    )
    
    if "Search" in mode:
        st.session_state["mode"] = "search"
    elif "Quiz" in mode:
        st.session_state["mode"] = "quiz"
    elif "Review" in mode:
        st.session_state["mode"] = "review"
    
    st.divider()
    
      # 가용 주제
    st.markdown("### 📚 현재 학습 가능한 주제")
    st.markdown("""
            - 시간 전치사 (at, on, in)
            - 장소 전치사 (at, on, in)
            - 소유격 (my, mine, his)
            - Does he? vs Is he?
            - 수동태 (be + 과거분사)
            - 현재완료
            - that 용법
            - I'm not used to 패턴
            - Do you? vs Are you?
    """)
    
    st.divider()
    
    # 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()
    
    st.divider()
    

# ============================================================
# 5. 메인 영역
# ============================================================

session_id = st.session_state["session_id"]
# 대화 기록 표시
history = get_session_history(session_id)
for msg in reversed(history.messages):
    role = "user" if msg.type == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)
        

# 사용자 입력    
user_input = st.chat_input("질문을 입력하세요...")
 
if user_input:      
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            
            # ============================================================
            # Search 모드
            # ============================================================
            if st.session_state["mode"] == "search":
                
                results_with_scores = vectorstore.similarity_search_with_score(user_input, k=3)
                # 관련도 필터링
                
                # 첫 번째 점수 확인
                if results_with_scores and results_with_scores[0][1] < 0.30:
                    # 관련 있음
                    relevant_results = [
                        (doc, score) for doc, score in results_with_scores 
                        if score < 0.6
                    ]
                    print(relevant_results)
                    if relevant_results:
                        relevant_docs = [doc for doc, score in relevant_results]
                        context = "\n\n".join([doc.page_content for doc in relevant_docs])
                        
                        # 답변 생성
                        response = chain_with_history.invoke(
                            {"context": context, 
                                "question": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )
                        
                        st.markdown("💡 **설명:**")
                        st.markdown(response.content)
                        
                        st.markdown("---")
                        st.markdown("📺 **관련 영상:**")
                        
                        for doc in relevant_docs:
                            url = doc.metadata['video_url']
                            start = int(float(doc.metadata.get("start_time", 0)))
                            # URL에서 video_id 추출
                            if "watch?v=" in url:
                                video_id = url.split("watch?v=")[-1].split("&")[0]
                            elif "youtu.be/" in url:
                                video_id = url.split("youtu.be/")[-1].split("?")[0]
                            else:
                                video_id = url  # fallback

                            embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}"
                            st.components.v1.iframe(embed_url, width=800, height=500)
                        st.session_state["context"] = context
                    else:
                        st.warning("관련 자료 없음")
                else:
                    # 관련 없음
                    st.warning("⚠️ 해당 내용은 강의 자료에 없습니다.")
                    with st.expander("📚 학습 가능한 주제", expanded=True):
                        st.markdown(AVAILABLE_TOPICS)
                                #st.link_button(
                                #    "▶️ 재생", 
                                #    f"{url}&t={start}s"
                                #)
                            
            # ============================================================
            # Quiz 모드
            # ============================================================
            elif st.session_state["mode"] == "quiz":
                history = get_session_history(session_id)
                past_text = "\n".join([m.content for m in history.messages if m.type == "human"])
                num_questions = 5
                context = st.session_state.get("context", "")
                quiz_prompt = ChatPromptTemplate.from_messages([
                        ("system", "너는 친절한 영어 선생님이야."),
                        ("human", """지금까지 사용자가 학습 중 물어본 내용은 다음과 같아:
                    {past_text}

                    이 내용을 참고해서 아래 {num_questions}개의 객관식 문제를 만들어줘 문제는 한글로.

                    {context}

                    출력 형식(JSON):
                    {{
                        "questions": [
                            {{
                                "question": "문제",
                                "options": ["1. 답1", "2. 답2", "3. 답3", "4. 답4"],
                                "answer": 1
                            }}
                        ]
                    }}
                    """)
                    ])
                
                quiz_chain = quiz_prompt | llm
                quiz = quiz_chain.invoke({
                    "past_text": past_text,  
                    "context": context,  
                    "num_questions": 5    
                })

                import re
                def safe_json_parse(text: str):
                    """LangChain 응답에서 JSON 본문만 안전하게 추출"""
                    if not text or not text.strip():
                        raise ValueError("빈 응답입니다.")
                    # 코드펜스 제거
                    if text.startswith("```"):
                        text = re.sub(r"^```(?:json)?", "", text)
                        text = re.sub(r"```$", "", text)
                    # JSON 블록만 추출
                    m = re.search(r"\{[\s\S]*\}", text)
                    if not m:
                        raise ValueError("JSON 객체를 찾을 수 없습니다.")
                    return json.loads(m.group(0))
                
                quiz_json = safe_json_parse(getattr(quiz, "content", ""))
                            
                st.title("🧩 영어 퀴즈")

                # 세션에 사용자 답안 저장
                if "user_answers" not in st.session_state:
                    st.session_state.user_answers = {}

                # 문제 렌더링
                for i, q in enumerate(quiz_json["questions"], 1):
                    st.markdown(f"**Q{i}. {q['question']}**")
                    selected = st.radio(
                        label="",
                        options=[f"{j+1}. {opt}" for j, opt in enumerate(q["options"])],
                        key=f"q{i}"
                    )
                    st.session_state.user_answers[i] = selected
            
            # ============================================================
            # Review 모드
            # ============================================================
            elif st.session_state["mode"] == "review":
                st.markdown("📖 **복습 자료**")
 