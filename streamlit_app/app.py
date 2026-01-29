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
from rapidfuzz import process, fuzz
import unicodedata
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
            persist_directory="../chroma_db",
            embedding_function=embeddings,
            )
    
    return vectorstore

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k":5})

# 지식 그래프 로드 (파일 수정 시간 기반 캐시)
@st.cache_resource
def load_knowledge_graph(_mtime):
    with open('/data/edutem/sooine/rag_bot/knowledge_graph.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# 파일 수정 시간이 바뀌면 캐시 무효화
kg_path = '/data/edutem/sooine/rag_bot/knowledge_graph.json'
kg_mtime = os.path.getmtime(kg_path)
knowledge_graph = load_knowledge_graph(kg_mtime)


# ============================================================
# 지식 그래프 검색 함수
# ============================================================
def rewrite_query(query):
    # 실제 토픽 목록을 프롬프트에 제공
    available_topics = list(knowledge_graph.keys())
    topics_str = ", ".join(available_topics)

    prompt = f"""사용자의 질문을 분석해서 아래 토픽 목록 중 정확히 일치하는 것을 선택하세요.
                사용 가능한 토픽 목록:
                {topics_str}

                사용자 질문: {query}

                규칙:
                - 위 목록에 있는 토픽 중에서만 선택하세요
                - 질문의 핵심 개념과 정확히 일치하는 토픽만 선택하세요
                - 비슷하지만 다른 개념이면 "없음"을 출력하세요 (예: "현재 시제" ≠ "현재진행형")
                - 목록에 정확히 일치하는 토픽이 없으면 반드시 "없음"을 출력하세요

                출력: 토픽 이름 또는 "없음" (설명 없이)"""
    return llm.invoke(prompt).content.strip()

def normalize(s):
    return unicodedata.normalize("NFC", s.lower().replace(" ", ""))

def fuzzy_match_topic(query, topic_list):
    #대충 비슷한 문자열도 매칭해주는 알고리즘
    q = normalize(query)
    candidates = [normalize(t) for t in topic_list]
    match, score, idx = process.extractOne(q, candidates, scorer=fuzz.ratio)
    if score > 70:  # threshold 조정 가능
        return topic_list[idx]
    return None

def search_in_knowledge_graph(query):
    """지식 그래프에서 키워드/질문 검색"""
    query_lower = query.lower().strip()
    topic_list = list(knowledge_graph.keys())

    # 1단계: 정확한 일치
    for main_topic in topic_list:
        if main_topic.lower() == query_lower:
            return {
                "type": "main_topic",
                "main_topic": main_topic,
                "data": knowledge_graph[main_topic]
            }

    # 2단계: 부분 일치
    for main_topic in topic_list:
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

    if best_match and max_score >= 1:
        return best_match

    # 4단계: Fuzzy match (마지막 수단)
    best = fuzzy_match_topic(query, topic_list)
    if best:
        return {
            "type": "main_topic",
            "main_topic": best,
            "data": knowledge_graph[best]
        }

    # 5단계: 매칭 실패
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
    
      # 가용 주제 (knowledge_graph에서 동적으로 로드)
    st.markdown("### 📚 현재 학습 가능한 주제")
    topic_list = list(knowledge_graph.keys())
    for topic in topic_list:
        st.markdown(f"- {topic}")
    
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
                rewritten = rewrite_query(user_input)

                # "없음"이면 지식 그래프 검색 스킵하고 벡터 검색으로
                kg_result = None
                if rewritten and rewritten != "없음":
                    kg_result = search_in_knowledge_graph(rewritten)

                if kg_result:
                    # ========== 지식 그래프에서 찾음 ==========
                    main_topic = kg_result['main_topic']
                    topic_data = kg_result['data'] if kg_result['type'] == 'main_topic' else knowledge_graph[main_topic]

                    # 0단계: 질문에서 핵심 키워드 추출 (매칭된 토픽이 아닌 원래 질문의 키워드)
                    keyword_prompt = f"""사용자 질문에서 핵심 문법 키워드만 추출하세요.

                            사용자 질문: {user_input}

                            규칙:
                            - 질문 형식 제거 (뭐야?, 알려줘, 설명해줘 등)
                            - 핵심 문법 용어만 추출
                            - 예: "if가 뭐야?" → "if"
                            - 예: "to 부정사 설명해줘" → "to 부정사"

                            출력: 키워드만 (설명 없이)"""
                    topic_keyword = llm.invoke(keyword_prompt).content.strip()

                    # 1단계: LLM이 일반적인 개념 설명 (키워드 기준, 다양한 용법 포함)
                    st.markdown(f"## 💡 {topic_keyword}")
                    with st.spinner("설명 생성 중..."):
                        explain_prompt = f"""'{topic_keyword}'이 무엇인지 핵심만 간단히 2-3문장으로 설명해주세요.
                         만약 '{topic_keyword}'이 여러 용법으로 쓰일 수 있다면 간단히 언급해주세요."""
                        explanation = llm.invoke(explain_prompt).content.strip()
                    st.write(explanation)

                    # 2단계: 해당 강의 영상 (RAG로 가장 관련성 높은 구간 찾기)
                    st.markdown("---")
                    st.success(f"✅ 이 교재에서 '{main_topic}'을(를) 다루고 있어요!")

                    # 벡터 검색으로 해당 토픽의 가장 관련성 높은 구간 찾기
                    rag_docs = retriever.invoke(main_topic)
                    if rag_docs:
                        st.markdown("### 📺 강의 영상")
                        best_doc = rag_docs[0]
                        url = best_doc.metadata.get('video_url', '')
                        start = int(float(best_doc.metadata.get('start_time', 0)))
                        end = int(float(best_doc.metadata.get('end_time', 0)))

                        if url:
                            if "watch?v=" in url:
                                video_id = url.split("watch?v=")[-1].split("&")[0]
                            elif "youtu.be/" in url:
                                video_id = url.split("youtu.be/")[-1].split("?")[0]
                            else:
                                video_id = url

                            embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                            st.components.v1.iframe(embed_url, width=800, height=450)

                    # 3단계: 관련 주제
                    st.markdown("---")
                    available_topics = list(knowledge_graph.keys())

                    # 3-1: 지식그래프의 related_topics (fuzzy 매칭)
                    kg_related_raw = topic_data.get('related_topics', [])
                    kg_related = []
                    for rel in kg_related_raw:
                        if rel == main_topic:
                            continue
                        # 정확히 있으면 그대로
                        if rel in knowledge_graph:
                            kg_related.append(rel)
                        else:
                            # fuzzy 매칭 시도
                            for kg_topic in available_topics:
                                if rel.lower() in kg_topic.lower() or kg_topic.lower() in rel.lower():
                                    if kg_topic != main_topic:
                                        kg_related.append(kg_topic)
                                    break

                    # 3-2: LLM 추천 관련 주제 (영어 문법 맥락 명시)
                    related_prompt = f"""'{main_topic}'과 관련된 영어 문법 주제 5개를 나열하세요.

규칙:
- 영어 문법/회화 관련 주제만 (프로그래밍 금지)
- 오직 주제 이름만 출력 (설명, 번호, 문장 금지)
- 콤마로 구분
- 예시 출력: 현재진행형, 현재완료, 과거시제, be동사, 조동사

출력:"""
                    llm_related_result = llm.invoke(related_prompt).content.strip()
                    llm_related = [t.strip() for t in llm_related_result.split(",")]

                    # 표시: 지식그래프 related_topics (RAG로 영상 구간 찾기)
                    if kg_related:
                        st.markdown("### 🔗 관련 주제 (교재 기준)")
                        for rel in kg_related:
                            with st.expander(f"✅ **{rel}** - 강의 있음", expanded=False):
                                # RAG로 해당 토픽의 영상 구간 찾기
                                rel_docs = retriever.invoke(rel)
                                if rel_docs:
                                    rel_best = rel_docs[0]
                                    url = rel_best.metadata.get('video_url', '')
                                    start = int(float(rel_best.metadata.get('start_time', 0)))
                                    end = int(float(rel_best.metadata.get('end_time', 0)))

                                    if url:
                                        if "watch?v=" in url:
                                            video_id = url.split("watch?v=")[-1].split("&")[0]
                                        elif "youtu.be/" in url:
                                            video_id = url.split("youtu.be/")[-1].split("?")[0]
                                        else:
                                            video_id = url

                                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                                        st.components.v1.iframe(embed_url, width=700, height=400)

                    # 표시: LLM 추천 관련 주제 (중복 포함, 전부 표시)
                    st.markdown("### 🤖 관련 주제 (AI 추천)")
                    shown_kg_topics = set()  # 이미 보여준 지식그래프 토픽 추적
                    for topic in llm_related:
                        if topic.lower() == main_topic.lower():
                            continue

                        # 지식 그래프에서 매칭 확인 (정확한 매칭 우선)
                        matched_kg_topic = None
                        # 1) 정확히 일치
                        for kg_topic in available_topics:
                            if topic.lower() == kg_topic.lower():
                                matched_kg_topic = kg_topic
                                break
                        # 2) 부분 일치 (정확한 매칭 없을 때만)
                        if not matched_kg_topic:
                            for kg_topic in available_topics:
                                if topic.lower() in kg_topic.lower() or kg_topic.lower() in topic.lower():
                                    matched_kg_topic = kg_topic
                                    break

                        if matched_kg_topic:
                            # 이미 보여준 토픽이면 스킵
                            if matched_kg_topic in shown_kg_topics:
                                continue
                            shown_kg_topics.add(matched_kg_topic)

                            with st.expander(f"✅ **{topic}** → {matched_kg_topic}", expanded=False):
                                # RAG로 해당 토픽의 영상 구간 찾기
                                rel_docs = retriever.invoke(matched_kg_topic)
                                if rel_docs:
                                    rel_best = rel_docs[0]
                                    url = rel_best.metadata.get('video_url', '')
                                    start = int(float(rel_best.metadata.get('start_time', 0)))
                                    end = int(float(rel_best.metadata.get('end_time', 0)))

                                    if url:
                                        if "watch?v=" in url:
                                            video_id = url.split("watch?v=")[-1].split("&")[0]
                                        elif "youtu.be/" in url:
                                            video_id = url.split("youtu.be/")[-1].split("?")[0]
                                        else:
                                            video_id = url

                                        embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                                        st.components.v1.iframe(embed_url, width=700, height=400)
                        else:
                            st.markdown(f"❌ **{topic}** - 해당 강의 없음")

                else:
                    # ========== 지식 그래프에 정확히 없음 → LLM 설명 + 관련 주제 연결 ==========

                    # 0단계: 질문에서 핵심 키워드 추출
                    keyword_prompt = f"""사용자 질문에서 핵심 문법 키워드만 추출하세요.

사용자 질문: {user_input}

규칙:
- 질문 형식 제거 (뭐야?, 알려줘, 설명해줘 등)
- 핵심 문법 용어만 추출
- 예: "현재시제가 뭐야?" → "현재시제"
- 예: "to 부정사 설명해줘" → "to 부정사"

출력: 키워드만 (설명 없이)"""
                    topic_keyword = llm.invoke(keyword_prompt).content.strip()

                    # 1단계: LLM이 질문에 대해 설명 (간단히)
                    with st.spinner("설명 생성 중..."):
                        explain_prompt = f"""'{topic_keyword}'이 무엇인지 핵심만 간단히 2-3문장으로 설명해주세요."""
                        explanation = llm.invoke(explain_prompt).content.strip()

                    st.markdown(f"## 💡 {topic_keyword}란 무엇일까요?")
                    st.write(explanation)

                    # 2단계: 이 교재에서는 해당 주제를 직접 다루지 않음을 알림
                    st.markdown("---")
                    st.warning(f"📭 이 교재에서는 '{topic_keyword}'을(를) 직접 다루는 강의는 없어요.")

                    # 3단계: 관련 주제 추천 (LLM이 문법적으로 연관된 주제들 나열)
                    related_prompt = f"""'{topic_keyword}'과 관련된 영어 문법 주제 5개를 나열하세요.

규칙:
- 영어 문법/회화 관련 주제만 (프로그래밍 금지)
- 오직 주제 이름만 출력 (설명, 번호, 문장 금지)
- 콤마로 구분
- 예시 출력: 현재진행형, 현재완료, 과거시제, be동사, 조동사

출력:"""
                    related_result = llm.invoke(related_prompt).content.strip()

                    if related_result:
                        related_topics = [t.strip() for t in related_result.split(",")]
                        available_topics = list(knowledge_graph.keys())

                        st.markdown("### 🔗 관련 주제")

                        for topic in related_topics:
                            # 지식 그래프에서 매칭 확인
                            matched_kg_topic = None
                            for kg_topic in available_topics:
                                if topic.lower() in kg_topic.lower() or kg_topic.lower() in topic.lower():
                                    matched_kg_topic = kg_topic
                                    break

                            if matched_kg_topic:
                                # 교재에 있음 → RAG로 영상 구간 찾기
                                with st.expander(f"✅ **{topic}** - 강의 있음", expanded=False):
                                    rel_docs = retriever.invoke(matched_kg_topic)
                                    if rel_docs:
                                        rel_best = rel_docs[0]
                                        url = rel_best.metadata.get('video_url', '')
                                        start = int(float(rel_best.metadata.get('start_time', 0)))
                                        end = int(float(rel_best.metadata.get('end_time', 0)))

                                        if url:
                                            if "watch?v=" in url:
                                                video_id = url.split("watch?v=")[-1].split("&")[0]
                                            elif "youtu.be/" in url:
                                                video_id = url.split("youtu.be/")[-1].split("?")[0]
                                            else:
                                                video_id = url

                                            embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}&end={end}"
                                            st.components.v1.iframe(embed_url, width=700, height=400)
                            else:
                                # 교재에 없음
                                st.markdown(f"❌ **{topic}** - 해당 강의 없음")            
              
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
#            elif st.session_state["mode"] == "quiz":
#                history = get_session_history(session_id)
#                past_text = "\n".join([m.content for m in history.messages if m.type == "human"])
#                num_questions = 5
#                context = st.session_state.get("context", "")
#                quiz_prompt = ChatPromptTemplate.from_messages([
#                        ("system", "너는 친절한 영어 선생님이야."),
#                        ("human", """지금까지 사용자가 학습 중 물어본 내용은 다음과 같아:
#                    {past_text}
#
#                    이 내용을 참고해서 아래 {num_questions}개의 객관식 문제를 만들어줘 문제는 한글로.
#
#                    {context}
#
#                    출력 형식(JSON):
#                    {{
#                        "questions": [
#                            {{
#                                "question": "문제",
#                                "options": ["1. 답1", "2. 답2", "3. 답3", "4. 답4"],
#                                "answer": 1
#                            }}
#                        ]
#                    }}
#                    """)
#                    ])
#                
#                quiz_chain = quiz_prompt | llm
#                quiz = quiz_chain.invoke({
#                    "past_text": past_text,  
#                    "context": context,  
#                    "num_questions": 5    
#                })
#
#                import re
#                def safe_json_parse(text: str):
#                    """LangChain 응답에서 JSON 본문만 안전하게 추출"""
#                    if not text or not text.strip():
#                        raise ValueError("빈 응답입니다.")
#                    # 코드펜스 제거
#                    if text.startswith("```"):
#                        text = re.sub(r"^```(?:json)?", "", text)
#                        text = re.sub(r"```$", "", text)
#                    # JSON 블록만 추출
#                    m = re.search(r"\{[\s\S]*\}", text)
#                    if not m:
#                        raise ValueError("JSON 객체를 찾을 수 없습니다.")
#                    return json.loads(m.group(0))
#                
#                quiz_json = safe_json_parse(getattr(quiz, "content", ""))
#                            
#                st.title("🧩 영어 퀴즈")
#
#                # 세션에 사용자 답안 저장
#                if "user_answers" not in st.session_state:
#                    st.session_state.user_answers = {}
#
#                # 문제 렌더링
#                for i, q in enumerate(quiz_json["questions"], 1):
#                    st.markdown(f"**Q{i}. {q['question']}**")
#                    selected = st.radio(
#                        label="",
#                        options=[f"{j+1}. {opt}" for j, opt in enumerate(q["options"])],
#                        key=f"q{i}"
#                    )
#                    st.session_state.user_answers[i] = selected
#            
#            # ============================================================
#            # Review 모드
#            # ============================================================
#            elif st.session_state["mode"] == "review":
#                st.markdown("📖 **복습 자료**")
# 