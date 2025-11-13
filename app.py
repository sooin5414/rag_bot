import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit.components.v1 as components
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
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
            )
    
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# 지식 그래프 로드
@st.cache_resource
def load_knowledge_graph():
    with open('/data/edutem/sooine/rag_bot/knowledge_graph.json', 'r', encoding='utf-8') as f:
        return json.load(f)

knowledge_graph = load_knowledge_graph()

AVAILABLE_TOPICS = """
현재 학습 가능한 주제:
1.
2.
3.
4.
5.
6.
7.
"""

# ============================================================
# 지식 그래프 검색 함수
# ============================================================

def search_in_knowledge_graph(query):
    """
    지식 그래프에서 키워드/질문 검색

    반환값:
    {
        "type": "main_topic" | "sub_topic" | None,
        "main_topic": "주제명",
        "sub_topic_id": "서브토픽 ID" (sub_topic인 경우),
        "data": {...}
    }
    """
    query_lower = query.lower().strip()

    # 1단계: Main topic 완전 일치
    for main_topic in knowledge_graph.keys():
        if main_topic.lower() == query_lower:
            return {
                "type": "main_topic",
                "main_topic": main_topic,
                "data": knowledge_graph[main_topic]
            }

    # 2단계: Main topic 부분 일치
    for main_topic in knowledge_graph.keys():
        if query_lower in main_topic.lower() or main_topic.lower() in query_lower:
            return {
                "type": "main_topic",
                "main_topic": main_topic,
                "data": knowledge_graph[main_topic]
            }

    # 3단계: Sub-topic 검색 (title, concept, examples)
    best_match = None
    max_score = 0

    for main_topic, topic_data in knowledge_graph.items():
        for sub_id, sub_data in topic_data['sub_topics'].items():
            score = 0

            # title 매칭
            if query_lower in sub_data['title'].lower():
                score += 3

            # concept 매칭
            if query_lower in sub_data['concept'].lower():
                score += 2

            # examples 매칭
            for example in sub_data.get('examples', []):
                if query_lower in example.lower():
                    score += 1
                    break

            if score > max_score:
                max_score = score
                best_match = {
                    "type": "sub_topic",
                    "main_topic": main_topic,
                    "sub_topic_id": sub_id,
                    "data": sub_data,
                    "score": score
                }

    # 매칭 점수가 1 이상이면 반환
    if best_match and max_score >= 1:
        return best_match

    # 4단계: 매칭 실패
    return None

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

llm = ChatOpenAI(model="gpt-4o", temperature=0)
prompt = ChatPromptTemplate.from_messages([
        ("system", """당신은 친절한 영어 학습 도우미입니다.
    
        주어진 컨텍스트를 바탕으로 질문에 답변하세요:
        {context}

        이전 대화 기록:
        {chat_history}

        답변은 친절하고 이해하기 쉽게 작성하세요."""),
            ("human", "{question}")
        ])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
    
# RAG 체인
rag_chain = (
    {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | prompt
    | llm
    | StrOutputParser()
)

def rag_with_chain(inputs):
    docs = retriever.invoke(inputs["question"])
    
    answer = (prompt | llm | StrOutputParser()).invoke({
        "context" : format_docs(docs),
        "question" : inputs["question"],
        "chat_history" : inputs["chat_history"]
    })
    return {"answer": answer, "source_docs": docs}


rag_chain = RunnableLambda(rag_with_chain) 

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
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

                # 1단계: 지식 그래프에서 먼저 검색
                kg_result = search_in_knowledge_graph(user_input)

                if kg_result:
                    # ========== 지식 그래프에서 찾음 ==========

                    if kg_result['type'] == 'main_topic':
                        # Main topic 전체 표시
                        main_topic = kg_result['main_topic']
                        topic_data = kg_result['data']

                        st.success(f"✅ 지식 그래프에서 찾았어요: **{main_topic}**")
                        st.markdown(f"## 📚 {main_topic}")
                        st.write(f"{main_topic}에 대해 알아볼까요?")

                        sub_topics = topic_data['sub_topics']

                        st.markdown("---")
                        st.markdown("### 🔍 세부 주제를 선택하세요")

                        # 각 서브토픽을 expander로 표시
                        for sub_id, sub_data in sub_topics.items():
                            with st.expander(f"▶️ {sub_data['title']}"):
                                st.write(sub_data['concept'])

                                # 예문
                                if sub_data.get('examples'):
                                    st.markdown("**예문:**")
                                    for ex in sub_data['examples']:
                                        st.write(f"- {ex}")

                                # 관련 영상
                                if sub_data.get('video_segments'):
                                    st.markdown("**📺 관련 영상:**")
                                    for idx, seg in enumerate(sub_data['video_segments'][:3]):  # 최대 3개
                                        url = seg['video_url']
                                        start = int(float(seg.get('start_time', 0)))
                                        end = int(float(seg.get('end_time', 0)))
                                        desc = seg.get('description', '')

                                        # video_id 추출
                                        if "watch?v=" in url:
                                            video_id = url.split("watch?v=")[-1].split("&")[0]
                                        elif "youtu.be/" in url:
                                            video_id = url.split("youtu.be/")[-1].split("?")[0]
                                        else:
                                            video_id = url  # fallback

                                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                                        st.components.v1.iframe(embed_url, width=800, height=450)

                    elif kg_result['type'] == 'sub_topic':
                        # Sub-topic만 집중 표시
                        main_topic = kg_result['main_topic']
                        sub_data = kg_result['data']

                        st.success(f"✅ 지식 그래프에서 찾았어요: **{main_topic}** > {sub_data['title']}")
                        st.markdown(f"## 📚 {sub_data['title']}")
                        st.write(f"**주제:** {main_topic}")

                        st.markdown("---")
                        st.markdown("### 💡 개념")
                        st.write(sub_data['concept'])

                        # 예문
                        if sub_data.get('examples'):
                            st.markdown("### 📝 예문")
                            for ex in sub_data['examples']:
                                st.write(f"- {ex}")

                        # 관련 영상
                        if sub_data.get('video_segments'):
                            st.markdown("---")
                            st.markdown("### 📺 관련 영상")
                            for idx, seg in enumerate(sub_data['video_segments'][:5]):  # 최대 5개
                                url = seg['video_url']
                                start = int(float(seg.get('start_time', 0)))
                                end = int(float(seg.get('end_time', 0)))
                                desc = seg.get('description', '')

                                st.write(f"**{idx+1}. {desc}** ({start}초 ~ {end}초)")

                                # video_id 추출
                                if "watch?v=" in url:
                                    video_id = url.split("watch?v=")[-1].split("&")[0]
                                elif "youtu.be/" in url:
                                    video_id = url.split("youtu.be/")[-1].split("?")[0]
                                else:
                                    video_id = url  # fallback

                                embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                                st.components.v1.iframe(embed_url, width=800, height=450)

                else:
                    # ========== 지식 그래프에 없음 → 벡터 검색 ==========
                    st.info("🔍 벡터 검색으로 답변을 생성합니다...")

                    with st.spinner("답변 생성 중..."):
                        result = chain_with_history.invoke(
                            {"question": user_input},
                            config={"configurable": {"session_id": session_id}}
                        )

                    st.markdown("### 💡 설명")
                    st.write(result["answer"])

                    st.markdown("---")
                    st.markdown("### 📺 관련 영상")

                    for i, doc in enumerate(result["source_docs"][:3], 1):
                        topic = doc.metadata.get('topic', '강의')
                        url = doc.metadata.get('video_url', '#')
                        start = int(float(doc.metadata.get("start_time", 0)))

                        st.write(f"**{i}. {topic}**")
                        st.write(doc.page_content[:200] + "...")

                        # URL에서 video_id 추출
                        if "watch?v=" in url:
                            video_id = url.split("watch?v=")[-1].split("&")[0]
                        elif "youtu.be/" in url:
                            video_id = url.split("youtu.be/")[-1].split("?")[0]
                        else:
                            video_id = url  # fallback

                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}"
                        st.components.v1.iframe(embed_url, width=800, height=500)

                        if url != '#':
                            video_url_with_time = f"{url}&t={start}s"
                            st.markdown(f"[▶️ 영상 보기 ({start}초)]({video_url_with_time})")            
              
                #st.session_state["context"] = context
            #else:
            #    st.warning("⚠️ 해당 내용은 강의 자료에 없습니다.")
            #    with st.expander("📚 학습 가능한 주제", expanded=True):
            #        st.markdown(AVAILABLE_TOPICS)
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
 