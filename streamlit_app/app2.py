"""
영어 학습 도우미 v2 - 하이브리드
- Knowledge Graph (26개) + Smart Search (113개 전체)
- KG에 없는 토픽도 Smart Search로 커버
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

st.set_page_config(page_title="영어 학습 도우미 v2", page_icon="📚", layout="wide")

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
# Smart Search 함수
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


def smart_search(query, k=5, threshold=0.35, rewritten=None):
    """Rewrite + 검색 + LLM 판단"""
    
    # 1. Rewrite (오타 교정)
    if rewritten is None:
        rewritten = rewrite_query(query)
    
    # 2. 검색
    results = vectorstore.similarity_search_with_score(rewritten, k=k)
    
    if not results:
        return {
            "found": False,
            "rewritten": rewritten,
            "keyword": rewritten,
            "message": "검색 결과가 없습니다.",
            "results": []
        }
    
    # 로그
    print(f"\n{'='*60}")
    print(f"Query: {query} → Rewritten: {rewritten}")
    for i, (doc, score) in enumerate(results):
        print(f"  {i}. [{score:.3f}] {doc.metadata.get('topic')}")
    
    # 3. 컨텍스트 생성 (토픽, 점수, 시간, 내용 포함)
    context = "\n".join([
        f"{i}. {doc.metadata.get('topic')} (score: {score:.3f}, start: {doc.metadata.get('start_time', 0):.1f}초): {doc.page_content[:100]}..."
        for i, (doc, score) in enumerate(results)
    ])

    # 4. LLM 판단
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": f"""영어 문법 검색 결과를 분석하세요.

질문: {query}
검색어: {rewritten}

검색 결과:
{context}

JSON 응답:
{{
    "found": true/false (질문과 관련된 결과가 있는지),
    "best_index": 가장 적합한 결과 번호 (0~{k-1}),
    "best_topic": "가장 적합한 토픽",
    "keyword": "핵심 문법 키워드",
    "message": "사용자에게 보여줄 메시지 (한국어)"
}}

선택 규칙 (순서대로 적용):
1. **토픽 정확도 우선**: 질문이 "~가 뭐야?" 같은 기본 개념 질문이면, 검색어와 정확히 일치하는 토픽을 선택
   - 예: "수동태가 뭐야?" → "수동태" (O), "수동태의 다양한 시제" (X - 세부 주제임)
   - 예: "be동사가 뭐야?" → "be동사" (O), "be동사의 부정문" (X)
2. **점수 고려**: 토픽이 같다면 score가 낮은 것 선택 (score가 낮을수록 관련성 높음)
3. **시간 고려**: 토픽과 점수가 비슷하면 start 시간이 빠른 것 선택
4. score > {threshold}이면 관련성 낮음 (found: false)"""
        }],
        response_format={"type": "json_object"},
        max_tokens=200,
        temperature=0
    )
    
    judgment = json.loads(response.choices[0].message.content)
    
    # 5. 결과 구성
    best_idx = min(judgment.get("best_index", 0), len(results) - 1)
    best_doc, best_score = results[best_idx]

    # Score threshold 체크 (단, LLM이 정확한 토픽 매칭을 찾았다면 약간의 여유 허용)
    best_topic = judgment.get("best_topic", "")
    if best_score > threshold:
        # LLM이 찾은 토픽이 rewritten과 정확히 일치하면 threshold를 약간 완화 (0.05 여유)
        if best_topic != rewritten or best_score > threshold + 0.05:
            judgment["found"] = False
    
    print(f"  → found: {judgment['found']}, best: {judgment.get('best_topic')}")
    
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

st.title("📚 영어 학습 도우미 v2")
st.markdown("하이브리드: Knowledge Graph + Smart Search")

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
        help="하이브리드: KG에서 먼저 찾고 없으면 Smart Search\nSmart Search만: 전체 113개 토픽 직접 검색"
    )

    st.divider()

    threshold = st.slider(
        "검색 민감도 (threshold)",
        min_value=0.20,
        max_value=0.50,
        value=0.35,
        step=0.05,
        help="낮을수록 엄격하게 '없음' 판정"
    )

    st.divider()

    # Knowledge Graph 주제 표시
    #st.markdown("### 📚 KG 주제")
    #st.caption(f"{len(knowledge_graph)}개")
    #with st.expander("주제 목록"):
    #    for topic in list(knowledge_graph.keys()):
    #        st.markdown(f"- {topic}")

    st.divider()

    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()

    st.divider()
    st.markdown("### 📊 시스템")
    st.markdown("- ChromaDB: 149개 토픽, 384 문서")
    st.markdown("- KG: 26개 주제")
    st.markdown("- LLM: GPT-4o")

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):

            if "하이브리드" in search_mode:
                # ========== 하이브리드 모드: KG 우선 → 없으면 Smart Search ==========
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

                    # Smart Search로 최적 영상 구간 찾기
                    search_result = smart_search(main_topic, threshold=threshold, rewritten=main_topic)
                    if search_result["found"]:
                        st.markdown("### 📺 강의 영상")
                        video_url = search_result.get("video_url")
                        start = int(float(search_result.get("start_time", 0)))
                        end = int(float(search_result.get("end_time", 0)))

                        embed_url = get_video_embed(video_url, start, end)
                        if embed_url:
                            st.components.v1.iframe(embed_url, width=800, height=450)
                            st.caption(f"⏱️ {start}초 ~ {end}초 | Score: {search_result['score']:.3f}")

                        # 관련 주제
                        st.markdown("---")
                        st.markdown("### 🔗 관련 주제")
                        shown = {search_result.get("best_topic", "")}
                        for doc, doc_score in search_result.get("results", [])[1:]:
                            rel_topic = doc.metadata.get("topic", "")
                            if rel_topic and rel_topic not in shown and doc_score <= threshold:
                                shown.add(rel_topic)
                                with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                    rel_url = doc.metadata.get('video_url', '')
                                    rel_start = int(float(doc.metadata.get('start_time', 0)))
                                    rel_end = int(float(doc.metadata.get('end_time', 0)))
                                    rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                    if rel_embed:
                                        st.components.v1.iframe(rel_embed, width=700, height=400)

                else:
                    # KG에 없음 → Smart Search로 전환
                    st.caption("⚡ KG에 없음 → Smart Search 사용")
                    result = smart_search(user_input, threshold=threshold, rewritten=rewritten)

                    keyword = result.get("keyword", user_input)
                    rewritten = result.get("rewritten", "")
                    score = result.get("score", 1.0)

                    if result["found"]:
                        # 찾았을 때
                        topic = result.get("best_topic", "")

                        st.markdown(f"## 💡 {keyword}")
                        st.caption(f"🔍 {user_input} → {rewritten} → {topic} (score: {score:.3f})")

                        st.success(f"✅ '{topic}' 강의가 있어요!")

                        # 영상
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

                        shown = {topic}
                        for doc, doc_score in result["results"][1:]:
                            rel_topic = doc.metadata.get("topic", "")
                            if rel_topic and rel_topic not in shown and doc_score <= threshold:
                                shown.add(rel_topic)
                                with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                    rel_url = doc.metadata.get('video_url', '')
                                    rel_start = int(float(doc.metadata.get('start_time', 0)))
                                    rel_end = int(float(doc.metadata.get('end_time', 0)))
                                    rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                    if rel_embed:
                                        st.components.v1.iframe(rel_embed, width=700, height=400)

                    else:
                        # 못 찾았을 때
                        st.markdown(f"## 💡 {keyword}")
                        st.caption(f"🔍 {user_input} → {rewritten} (score: {score:.3f})")

                        st.warning(f"📭 '{keyword}'을(를) 직접 다루는 강의가 없어요.")

                        # 가장 가까운 토픽 제안
                        if result.get("results"):
                            closest = result["results"][0][0].metadata.get("topic", "")
                            if closest:
                                st.info(f"💡 가장 가까운 토픽: **{closest}**")

            else:
                # ========== Smart Search만 사용 ==========
                result = smart_search(user_input, threshold=threshold)

                keyword = result.get("keyword", user_input)
                rewritten = result.get("rewritten", "")
                score = result.get("score", 1.0)

                if result["found"]:
                    topic = result.get("best_topic", "")

                    st.markdown(f"## 💡 {keyword}")
                    st.caption(f"🔍 {user_input} → {rewritten} → {topic} (score: {score:.3f})")

                    st.success(f"✅ '{topic}' 강의가 있어요!")

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

                    shown = {topic}
                    for doc, doc_score in result["results"][1:]:
                        rel_topic = doc.metadata.get("topic", "")
                        if rel_topic and rel_topic not in shown and doc_score <= threshold:
                            shown.add(rel_topic)
                            with st.expander(f"✅ {rel_topic} (score: {doc_score:.3f})"):
                                rel_url = doc.metadata.get('video_url', '')
                                rel_start = int(float(doc.metadata.get('start_time', 0)))
                                rel_end = int(float(doc.metadata.get('end_time', 0)))
                                rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                                if rel_embed:
                                    st.components.v1.iframe(rel_embed, width=700, height=400)
                else:
                    st.markdown(f"## 💡 {keyword}")
                    st.caption(f"🔍 {user_input} → {rewritten} (score: {score:.3f})")

                    st.warning(f"📭 '{keyword}'을(를) 직접 다루는 강의가 없어요.")

                    if result.get("results"):
                        closest = result["results"][0][0].metadata.get("topic", "")
                        if closest:
                            st.info(f"💡 가장 가까운 토픽: **{closest}**")

