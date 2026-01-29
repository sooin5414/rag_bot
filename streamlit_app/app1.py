"""
영어 학습 도우미 v3 - Confidence 기반 개선
- Knowledge Graph (26개) + Smart Search v2 (113개 전체)
- Threshold 제거, Confidence 기반 판단
- LLM Judge 강화
"""

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from openai import OpenAI
from dotenv import load_dotenv
from rapidfuzz import process, fuzz
import unicodedata
import json
import os

load_dotenv()

st.set_page_config(page_title="영어 학습 도우미 v3", page_icon="📚", layout="wide")

# ============================================================
# 벡터스토어 로드
# ============================================================

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = Chroma(
        persist_directory="/data/edutem/sooine/rag_bot/chroma_db",
        embedding_function=embeddings,
    )
    return vectorstore

vectorstore = load_vectorstore()
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Knowledge Graph 로드
@st.cache_resource
def load_knowledge_graph(_mtime):
    with open('/data/edutem/sooine/rag_bot/knowledge_graph.json', 'r', encoding='utf-8') as f:
        return json.load(f)

kg_path = '/data/edutem/sooine/rag_bot/knowledge_graph.json'
kg_mtime = os.path.getmtime(kg_path)
knowledge_graph = load_knowledge_graph(kg_mtime)

# ============================================================
# Knowledge Graph 검색 함수
# ============================================================

def normalize(s):
    return unicodedata.normalize("NFC", s.lower().replace(" ", ""))

def fuzzy_match_topic(query, topic_list):
    q = normalize(query)
    candidates = [normalize(t) for t in topic_list]
    match_result, score, idx = process.extractOne(q, candidates, scorer=fuzz.ratio)
    if score > 70:
        return topic_list[idx]
    return None

def search_in_knowledge_graph(query):
    """지식 그래프에서 검색"""
    query_lower = query.lower().strip()
    topic_list = list(knowledge_graph.keys())

    # 1단계: 정확한 일치
    for main_topic in topic_list:
        if main_topic.lower() == query_lower:
            return {"type": "main_topic", "main_topic": main_topic, "data": knowledge_graph[main_topic]}

    # 2단계: 부분 일치
    for main_topic in topic_list:
        if query_lower in main_topic.lower() or main_topic.lower() in query_lower:
            return {"type": "main_topic", "main_topic": main_topic, "data": knowledge_graph[main_topic]}

    # 3단계: Fuzzy match
    best = fuzzy_match_topic(query, topic_list)
    if best:
        return {"type": "main_topic", "main_topic": best, "data": knowledge_graph[best]}

    return None

# ============================================================
# Query Rewriting
# ============================================================

def rewrite_query(query):
    """오타 교정 + 문법 토픽 추출"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""사용자 질문에서 영어 문법 토픽을 추출하세요.

사용자 질문: {query}

규칙:
- 핵심 문법 용어만 추출
- 오타 교정 (비동사 → be동사, 머야 → 뭐야)
- 질문 형식 제거
- 예: "비동사가 머야?" → "be동사"
- 예: "have pp 어떻게 써?" → "현재완료"

출력: 토픽만 (설명 없이)"""
        }],
        max_tokens=50,
        temperature=0
    )
    return response.choices[0].message.content.strip()

# ============================================================
# 컨텍스트 포맷팅 함수
# ============================================================

def format_results_with_metadata(results):
    """
    검색 결과를 LLM이 이해하기 좋은 포맷으로 변환
    
    Args:
        results: List of (Document, score) or List of Document
    
    Returns:
        str: 포맷팅된 컨텍스트 문자열
    """
    if not results:
        return "검색 결과가 없습니다."
    
    context_parts = []
    
    for i, item in enumerate(results):
        # score 있는지 확인
        if isinstance(item, tuple):
            doc, score = item
        else:
            doc = item
            score = None
        
        # 메타데이터 추출
        topic = doc.metadata.get('topic', '알 수 없음')
        start_time = doc.metadata.get('start_time', 0)
        end_time = doc.metadata.get('end_time', 0)
        video_url = doc.metadata.get('video_url', '')
        
        # 시간 포맷 (MM:SS)
        start_str = f"{int(start_time//60)}:{int(start_time%60):02d}"
        end_str = f"{int(end_time//60)}:{int(end_time%60):02d}"
        duration = end_time - start_time
        
        # 내용 미리보기 (200자)
        content = doc.page_content.replace('\n', ' ')
        if len(content) > 200:
            content_preview = content[:200].rsplit(' ', 1)[0] + "..."
        else:
            content_preview = content
        
        # 포맷팅
        formatted = f"""
[결과 {i}]
- 토픽: {topic}
- 시간: {start_str} ~ {end_str} ({duration:.0f}초)
{"- 유사도: " + f"{score:.4f}" if score is not None else ""}
- 내용: {content_preview}
"""
        context_parts.append(formatted.strip())
    
    return "\n\n".join(context_parts)

# ============================================================
# Smart Search v2 (개선된 버전)
# ============================================================

def smart_search_v2(query, k=10, rewritten=None):
    """
    Confidence 기반 검색 (Threshold 제거)
    
    Args:
        query: 사용자 질문
        k: 검색할 문서 수
        rewritten: 미리 rewrite된 쿼리 (옵션)
    
    Returns:
        dict: 검색 결과와 confidence 정보
    """
    
    # 1. Rewrite (오타 교정)
    if rewritten is None:
        rewritten = rewrite_query(query)
    
    # 2. 검색 (score 포함)
    results = vectorstore.similarity_search_with_score(rewritten, k=k)
    
    if not results:
        return {
            "found": False,
            "confidence": "none",
            "rewritten": rewritten,
            "message": "검색 결과가 없습니다.",
            "results": []
        }
    
    # 로그 출력
    print(f"\n{'='*60}")
    print(f"Query: {query} → Rewritten: {rewritten}")
    for i, (doc, score) in enumerate(results):
        print(f"  {i}. [{score:.3f}] {doc.metadata.get('topic')}")
    
    # 3. 컨텍스트 생성
    context = format_results_with_metadata(results)
    
    # 4. LLM 판단 (개선된 프롬프트)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """당신은 영어 문법 교육 영상 검색 시스템의 판단자입니다.
검색 결과를 분석하여 사용자 질문에 가장 적합한 영상을 선택하세요.

판단 기준:
1. **정확한 일치** (high confidence): 질문의 핵심 토픽과 검색 결과 토픽이 정확히 일치
   예: "be동사가 뭐야?" → "be동사" 토픽
   
2. **유사 토픽** (medium confidence): 직접 일치는 없지만 답변 가능한 관련 토픽
   예: "수동태 시제" 질문 → "수동태" 토픽 (기본 개념으로 답변 가능)
   
3. **관련 없음** (low confidence): 질문과 관련성이 낮거나 너무 동떨어짐
   예: "가정법" 질문 → "현재완료" 토픽

**중요**: 유사도 점수는 참고만 하고, 토픽명과 내용을 우선적으로 평가하세요.
**토픽 선택**: 질문이 "~가 뭐야?" 같은 기본 개념 질문이면, 검색어와 정확히 일치하는 기본 토픽을 선택하세요."""
            },
            {
                "role": "user",
                "content": f"""
질문: {query}
검색어: {rewritten}

검색 결과:
{context}

다음 JSON 형식으로 답변하세요:
{{
    "confidence": "high" | "medium" | "low",
    "found": true | false,
    "best_index": 0부터 {k-1} 사이의 숫자,
    "best_topic": "선택한 토픽명",
    "reasoning": "선택 이유 (한국어 2-3문장)",
    "alternative_topics": ["관련있을 수 있는 다른 토픽들 (최대 3개)"]
}}
"""
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    
    judgment = json.loads(response.choices[0].message.content)
    
    # 5. 결과 구성
    best_idx = judgment.get("best_index", 0)
    if best_idx is None or best_idx >= len(results):
        best_idx = 0
    
    best_doc, best_score = results[best_idx]
    
    # confidence가 low면 found를 false로
    if judgment.get("confidence") == "low":
        judgment["found"] = False
    
    print(f"  → confidence: {judgment.get('confidence')}, found: {judgment.get('found')}, best: {judgment.get('best_topic')}")
    
    return {
        **judgment,
        "rewritten": rewritten,
        "score": best_score,
        "doc": best_doc,
        "video_url": best_doc.metadata.get('video_url'),
        "start_time": best_doc.metadata.get('start_time'),
        "end_time": best_doc.metadata.get('end_time'),
        "results": results
    }

# ============================================================
# 영상 임베드 URL 생성
# ============================================================

def get_video_embed(url, start, end):
    """YouTube 임베드 URL 생성"""
    if not url:
        return None
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    else:
        return None
    return f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}"

# ============================================================
# UI
# ============================================================

st.title("📚 영어 학습 도우미 v3")
st.markdown("**개선**: Confidence 기반 판단 (Threshold 제거)")

# 세션 초기화
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    st.header("⚙️ 설정")

    # 검색 모드 선택
    search_mode = st.radio(
        "검색 모드",
        ["🔍 하이브리드 (KG 우선)", "⚡ Smart Search만"],
        index=0,
        help="하이브리드: KG에서 먼저 찾고 없으면 Smart Search v2\nSmart Search만: 전체 113개 토픽 직접 검색"
    )

    st.divider()

    # k 값 조정
    k_value = st.slider(
        "검색 문서 수 (k)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="검색할 문서 수 (많을수록 정확하지만 느림)"
    )

    st.divider()

    # Knowledge Graph 주제 표시
    st.markdown("### 📚 KG 주제")
    st.caption(f"{len(knowledge_graph)}개")
    with st.expander("주제 목록 보기"):
        for topic in sorted(knowledge_graph.keys()):
            st.markdown(f"- {topic}")

    st.divider()

    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.markdown("### 📊 시스템")
    st.markdown("- ChromaDB: 149개 토픽, 384 문서")
    st.markdown("- KG: 26개 주제")
    st.markdown("- LLM: GPT-4o")
    st.markdown("- 버전: v3 (Confidence 기반)")

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):

            if "하이브리드" in search_mode:
                # ========== 하이브리드 모드: KG 우선 → 없으면 Smart Search v2 ==========
                rewritten = rewrite_query(user_input) 
                kg_result = search_in_knowledge_graph(rewritten)

                if kg_result:
                    # KG에서 찾음
                    st.caption("🔍 Knowledge Graph에서 찾음")
                    main_topic = kg_result['main_topic']

                    # LLM 설명
                    st.markdown(f"## 💡 {main_topic}")
                    with st.spinner("설명 생성 중..."):
                        explain_prompt = f"'{main_topic}'이 무엇인지 핵심만 간단히 2-3문장으로 설명해주세요."
                        explanation = llm.invoke(explain_prompt).content.strip()
                    st.write(explanation)

                    st.markdown("---")
                    st.success(f"✅ '{main_topic}' 강의가 있어요!")

                    # Smart Search v2로 최적 영상 구간 찾기
                    search_result = smart_search_v2(main_topic, k=k_value, rewritten=main_topic)
                    
                    confidence = search_result.get("confidence", "low")
                    
                    if confidence in ["high", "medium"]:
                        st.markdown("### 📺 강의 영상")
                        st.caption(f"🎯 확신도: {confidence} | Score: {search_result.get('score', 0):.3f}")
                        
                        video_url = search_result.get("video_url")
                        start = int(float(search_result.get("start_time", 0)))
                        end = int(float(search_result.get("end_time", 0)))

                        embed_url = get_video_embed(video_url, start, end)
                        if embed_url:
                            st.components.v1.iframe(embed_url, width=800, height=450)
                            st.caption(f"⏱️ {start}초 ~ {end}초")
                            
                        if search_result.get("reasoning"):
                            with st.expander("💡 선택 이유"):
                                st.write(search_result["reasoning"])

                        # 관련 주제
                        st.markdown("---")
                        st.markdown("### 🔗 관련 주제")
                        shown = {search_result.get("best_topic", "")}
                        for doc, doc_score in search_result.get("results", [])[1:6]:
                            rel_topic = doc.metadata.get("topic", "")
                            if rel_topic and rel_topic not in shown:
                                shown.add(rel_topic)
                                with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                    rel_url = doc.metadata.get('video_url', '')
                                    rel_start = int(float(doc.metadata.get('start_time', 0)))
                                    rel_end = int(float(doc.metadata.get('end_time', 0)))
                                    rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                    if rel_embed:
                                        st.components.v1.iframe(rel_embed, width=700, height=400)

                else:
                    # KG에 없음 → Smart Search v2로 전환
                    st.caption("⚡ KG에 없음 → Smart Search v2 사용")
                    result = smart_search_v2(user_input, k=k_value, rewritten=rewritten)

                    keyword = result.get("best_topic", user_input)
                    rewritten_query = result.get("rewritten", "")
                    score = result.get("score", 1.0)
                    confidence = result.get("confidence", "low")

                    st.markdown(f"## 💡 {keyword}")
                    st.caption(f"🔍 {user_input} → {rewritten_query} → {keyword}")
                    
                    if confidence == "high":
                        st.success(f"✅ '{keyword}' 강의를 찾았어요!")
                        st.caption(f"🎯 확신도: 높음 | Score: {score:.3f}")
                        
                    elif confidence == "medium":
                        st.info(f"🟡 '{keyword}' 토픽이 관련있을 수 있어요")
                        st.caption(f"⚠️ 확신도: 중간 | Score: {score:.3f}")
                        
                    else:  # low
                        st.warning(f"📭 '{keyword}'을(를) 직접 다루는 강의는 없어요")
                        st.caption(f"❓ 확신도: 낮음 | Score: {score:.3f}")
                    
                    # 선택 이유 표시
                    if result.get("reasoning"):
                        with st.expander("💡 판단 근거"):
                            st.write(result["reasoning"])

                    # 영상 표시 (high, medium일 때만)
                    if confidence in ["high", "medium"]:
                        video_url = result.get("video_url")
                        start = int(float(result.get("start_time", 0)))
                        end = int(float(result.get("end_time", 0)))

                        if video_url:
                            st.markdown("### 📺 강의 영상")
                            embed_url = get_video_embed(video_url, start, end)
                            if embed_url:
                                st.components.v1.iframe(embed_url, width=800, height=450)
                                st.caption(f"⏱️ {start}초 ~ {end}초")

                        # 관련 주제
                        st.markdown("---")
                        st.markdown("### 🔗 관련 주제")
                        shown = {keyword}
                        for doc, doc_score in result["results"][1:6]:
                            rel_topic = doc.metadata.get("topic", "")
                            if rel_topic and rel_topic not in shown:
                                shown.add(rel_topic)
                                with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                    rel_url = doc.metadata.get('video_url', '')
                                    rel_start = int(float(doc.metadata.get('start_time', 0)))
                                    rel_end = int(float(doc.metadata.get('end_time', 0)))
                                    rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                    if rel_embed:
                                        st.components.v1.iframe(rel_embed, width=700, height=400)
                    
                    # 대체 토픽 제안 (low confidence일 때)
                    if confidence == "low" and result.get("alternative_topics"):
                        st.markdown("### 🔍 이런 토픽들을 찾아보시겠어요?")
                        for alt in result["alternative_topics"]:
                            st.markdown(f"- {alt}")

            else:
                # ========== Smart Search v2만 사용 ==========
                result = smart_search_v2(user_input, k=k_value)

                keyword = result.get("best_topic", user_input)
                rewritten = result.get("rewritten", "")
                score = result.get("score", 1.0)
                confidence = result.get("confidence", "low")

                st.markdown(f"## 💡 {keyword}")
                st.caption(f"🔍 {user_input} → {rewritten} → {keyword}")

                if confidence == "high":
                    st.success(f"✅ '{keyword}' 강의를 찾았어요!")
                    st.caption(f"🎯 확신도: 높음 | Score: {score:.3f}")
                    
                elif confidence == "medium":
                    st.info(f"🟡 '{keyword}' 토픽이 관련있을 수 있어요")
                    st.caption(f"⚠️ 확신도: 중간 | Score: {score:.3f}")
                    
                else:  # low
                    st.warning(f"📭 '{keyword}'을(를) 직접 다루는 강의는 없어요")
                    st.caption(f"❓ 확신도: 낮음 | Score: {score:.3f}")
                
                # 선택 이유
                if result.get("reasoning"):
                    with st.expander("💡 판단 근거"):
                        st.write(result["reasoning"])

                # 영상 (high, medium만)
                if confidence in ["high", "medium"]:
                    video_url = result.get("video_url")
                    start = int(float(result.get("start_time", 0)))
                    end = int(float(result.get("end_time", 0)))

                    if video_url:
                        st.markdown("### 📺 강의 영상")
                        embed_url = get_video_embed(video_url, start, end)
                        if embed_url:
                            st.components.v1.iframe(embed_url, width=800, height=450)
                            st.caption(f"⏱️ {start}초 ~ {end}초")

                    st.markdown("---")
                    st.markdown("### 🔗 관련 주제")
                    shown = {keyword}
                    for doc, doc_score in result["results"][1:6]:
                        rel_topic = doc.metadata.get("topic", "")
                        if rel_topic and rel_topic not in shown:
                            shown.add(rel_topic)
                            with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                rel_url = doc.metadata.get('video_url', '')
                                rel_start = int(float(doc.metadata.get('start_time', 0)))
                                rel_end = int(float(doc.metadata.get('end_time', 0)))
                                rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                if rel_embed:
                                    st.components.v1.iframe(rel_embed, width=700, height=400)
                
                # 대체 토픽 제안
                if confidence == "low" and result.get("alternative_topics"):
                    st.markdown("### 🔍 이런 토픽들을 찾아보시겠어요?")
                    for alt in result["alternative_topics"]:
                        st.markdown(f"- {alt}")