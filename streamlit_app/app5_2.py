"""
영어 학습 도우미 v3.1 - Topic Routing + GPU Reranker
- KG (26) + Smart Search (전체)
- 2-stage retrieval: 토픽 확정 시 topic filter로 구간만 탐색
- CrossEncoder reranker (GPU)로 "정확한 지점" 선택 강화
- Chroma score는 dist(거리)로 취급
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
import re
from typing import Optional, List, Tuple, Dict, Any

load_dotenv()

# =========================
# Config
# =========================
OPENAI_REWRITE_MODEL = os.getenv("OPENAI_REWRITE_MODEL", "gpt-4o")
OPENAI_JUDGE_MODEL   = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")
OPENAI_EXPLAIN_MODEL = os.getenv("OPENAI_EXPLAIN_MODEL", "gpt-4o")

USE_RERANKER = os.getenv("USE_RERANKER", "1") == "1"
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# retrieve 후보 수: UI k_value(10)라면 2x=20 정도만 rerank
RETRIEVE_MULTIPLIER = float(os.getenv("RETRIEVE_MULTIPLIER", "2.0"))
RETRIEVE_MAX = int(os.getenv("RETRIEVE_MAX", "30"))

# E5 prefix/normalize 옵션 (기존 DB가 prefix 없이 구축됐으면, 최적 성능 위해 재색인 권장)
USE_E5_PREFIX = os.getenv("USE_E5_PREFIX", "0") == "1"

st.set_page_config(page_title="영어 학습 도우미 v3.1", page_icon="📚", layout="wide")

# =========================
# Utils
# =========================
def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s.lower().replace(" ", ""))

def safe_float(x, default=0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default

def is_basic_definition_question(q: str) -> bool:
    # "~가 뭐야", "뜻", "정의", "what is", "meaning" 류
    qn = q.strip().lower()
    patterns = [
        r"뭐야\??$", r"무슨뜻\??$", r"뜻이 뭐야\??$", r"정의\??$", r"설명해줘\??$",
        r"what is\b", r"meaning\b", r"define\b",
    ]
    return any(re.search(p, qn) for p in patterns)

def fuzzy_match_topic(query: str, topic_list: List[str], threshold: int = 85) -> Optional[str]:
    q = normalize(query)
    candidates = [normalize(t) for t in topic_list]
    mr = process.extractOne(q, candidates, scorer=fuzz.ratio)
    if not mr:
        return None
    _match, score, idx = mr
    if score >= threshold:
        return topic_list[idx]
    return None

# =========================
# Vectorstore (Chroma) load
# =========================
class E5Embeddings(HuggingFaceEmbeddings):
    def embed_query(self, text: str):
        if USE_E5_PREFIX:
            text = "query: " + text
        return super().embed_query(text)

    def embed_documents(self, texts):
        if USE_E5_PREFIX:
            texts = ["passage: " + t for t in texts]
        return super().embed_documents(texts)

@st.cache_resource
def load_vectorstore():
    embeddings = E5Embeddings(
        model_name="intfloat/multilingual-e5-large",
        encode_kwargs={"normalize_embeddings": True} if USE_E5_PREFIX else {},
    )
    vectorstore = Chroma(
        persist_directory="/data/edutem/sooine/rag_bot/chroma_db",
        embedding_function=embeddings,
    )
    return vectorstore

vectorstore = load_vectorstore()

@st.cache_resource
def load_all_topics_from_chroma() -> List[str]:
    # small corpus(384 docs)이면 충분히 감당됨
    metas = vectorstore._collection.get(include=["metadatas"])["metadatas"]
    topics = []
    for m in metas:
        if not m:
            continue
        t = m.get("topic")
        if t:
            topics.append(t)
    # unique, stable order
    seen = set()
    uniq = []
    for t in topics:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

ALL_TOPICS = load_all_topics_from_chroma()

client = OpenAI()
llm = ChatOpenAI(model=OPENAI_EXPLAIN_MODEL, temperature=0)

# =========================
# KG load
# =========================
@st.cache_resource
def load_knowledge_graph(_mtime):
    with open('/data/edutem/sooine/rag_bot/knowledge_graph.json', 'r', encoding='utf-8') as f:
        return json.load(f)

kg_path = '/data/edutem/sooine/rag_bot/knowledge_graph.json'
kg_mtime = os.path.getmtime(kg_path)
knowledge_graph = load_knowledge_graph(kg_mtime)

def search_in_knowledge_graph(query: str) -> Optional[Dict[str, Any]]:
    query_lower = query.lower().strip()
    topic_list = list(knowledge_graph.keys())

    # exact
    for main_topic in topic_list:
        if main_topic.lower() == query_lower:
            return {"type": "main_topic", "main_topic": main_topic, "data": knowledge_graph[main_topic]}

    # substring
    for main_topic in topic_list:
        if query_lower in main_topic.lower() or main_topic.lower() in query_lower:
            return {"type": "main_topic", "main_topic": main_topic, "data": knowledge_graph[main_topic]}

    # fuzzy
    best = fuzzy_match_topic(query, topic_list, threshold=80)
    if best:
        return {"type": "main_topic", "main_topic": best, "data": knowledge_graph[best]}

    return None

# =========================
# Query rewrite
# =========================
@st.cache_data(show_spinner=False)
def rewrite_query(query: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_REWRITE_MODEL,
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

# =========================
# Reranker (GPU)
# =========================
@st.cache_resource
def load_reranker():
    if not USE_RERANKER:
        return None
    try:
        import torch
        from sentence_transformers import CrossEncoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # max_length 줄이면 속도 안정화
        reranker = CrossEncoder(RERANKER_MODEL, device=device, max_length=512)
        return reranker
    except Exception as e:
        # sentence-transformers 미설치 등
        return None

RERANKER = load_reranker()

def rerank(query: str, docs_with_dist: List[Tuple[Any, float]]) -> List[Tuple[Any, float, float]]:
    """
    Args:
      docs_with_dist: [(doc, dist), ...]
    Returns:
      [(doc, dist, rerank_score), ...] sorted by rerank_score desc
    """
    if not RERANKER or not docs_with_dist:
        return [(d, dist, 0.0) for d, dist in docs_with_dist]

    pairs = [(query, d.page_content) for d, _dist in docs_with_dist]
    scores = RERANKER.predict(pairs)
    packed = [(docs_with_dist[i][0], docs_with_dist[i][1], float(scores[i])) for i in range(len(docs_with_dist))]
    packed.sort(key=lambda x: x[2], reverse=True)
    return packed

# =========================
# Context formatting for judge
# =========================
def format_results_with_metadata(results: List[Tuple[Any, float, float]]) -> str:
    """
    results: [(doc, dist, rerank_score), ...]
    """
    if not results:
        return "검색 결과가 없습니다."

    parts = []
    for i, (doc, dist, rr) in enumerate(results):
        topic = doc.metadata.get('topic', '알 수 없음')
        start_time = safe_float(doc.metadata.get('start_time', 0))
        end_time = safe_float(doc.metadata.get('end_time', 0))
        video_url = doc.metadata.get('video_url', '')

        start_str = f"{int(start_time//60)}:{int(start_time%60):02d}"
        end_str = f"{int(end_time//60)}:{int(end_time%60):02d}"
        duration = max(0.0, end_time - start_time)

        content = (doc.page_content or "").replace("\n", " ")
        content_preview = content[:240] + ("..." if len(content) > 240 else "")

        parts.append(
            f"""[후보 {i}]
- 토픽: {topic}
- 시간: {start_str} ~ {end_str} ({duration:.0f}초)
- dist(작을수록 유사): {dist:.4f}
- rerank(클수록 적합): {rr:.4f}
- 내용: {content_preview}"""
        )
    return "\n\n".join(parts)

def build_alternative_topics(results: List[Tuple[Any, float, float]], max_n: int = 3) -> List[str]:
    seen = set()
    alts = []
    for doc, _dist, _rr in results:
        t = doc.metadata.get("topic", "")
        if t and t not in seen:
            seen.add(t)
            alts.append(t)
        if len(alts) >= max_n:
            break
    return alts

# =========================
# YouTube embed
# =========================
def get_video_embed(url: str, start: float, end: float) -> Optional[str]:
    if not url:
        return None
    if "watch?v=" in url:
        video_id = url.split("watch?v=")[-1].split("&")[0]
    elif "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0]
    else:
        return None
    return f"https://www.youtube.com/embed/{video_id}?start={int(start)}&end={int(end)}"

# =========================
# Topic routing + search
# =========================
def resolve_topic(rewritten: str) -> Optional[str]:
    # exact 먼저
    if rewritten in ALL_TOPICS:
        return rewritten
    # fuzzy로 토픽 정규화
    best = fuzzy_match_topic(rewritten, ALL_TOPICS, threshold=88)
    return best

def retrieve_candidates(query: str, rewritten: str, k_ui: int, pinned_topic: Optional[str] = None):
    k_retrieve = min(int(k_ui * RETRIEVE_MULTIPLIER), RETRIEVE_MAX)

    # 1) 토픽이 pin되었거나 resolve되면: topic filter 검색 (구간 정밀)
    topic = pinned_topic or resolve_topic(rewritten)
    if topic:
        # 토픽 내부에서 "원 질문"으로 구간을 찾는 게 더 잘 맞음
        raw = vectorstore.similarity_search_with_score(query, k=k_retrieve, filter={"topic": topic})
        return topic, raw

    # 2) 토픽 확정 불가: 전역 검색은 rewritten으로 (문법 용어 중심)
    raw = vectorstore.similarity_search_with_score(rewritten, k=k_retrieve)
    return None, raw

def judge_best(query: str, rewritten: str, candidates: List[Tuple[Any, float, float]], k_ui: int) -> Dict[str, Any]:
    """
    candidates: reranked list [(doc, dist, rr), ...] already sorted by rr desc
    """
    if not candidates:
        return {
            "confidence": "low",
            "found": False,
            "best_index": 0,
            "best_topic": "",
            "reasoning": "검색 후보가 없어서 관련 영상을 선택할 수 없습니다.",
            "alternative_topics": []
        }

    context = format_results_with_metadata(candidates[:k_ui])

    response = client.chat.completions.create(
        model=OPENAI_JUDGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": """당신은 영어 문법 교육 영상 검색 시스템의 판단자입니다.
후보 구간들을 보고 사용자 질문에 가장 적합한 구간을 선택하세요.

중요:
- dist는 거리(distance)이며, 작을수록 유사합니다.
- rerank 점수는 클수록 질문 적합도가 높습니다.
- 점수는 참고용이고, 토픽/내용 적합도를 최우선으로 봅니다.

판단:
1) high: 질문 토픽과 후보 토픽/내용이 직접적으로 정확히 일치
2) medium: 직접 일치는 아니지만 충분히 답변 가능한 인접 토픽
3) low: 관련성이 낮음
"""
            },
            {
                "role": "user",
                "content": f"""
질문: {query}
검색어(토픽 추출): {rewritten}

후보:
{context}

다음 JSON 형식으로 답변:
{{
  "confidence": "high" | "medium" | "low",
  "found": true | false,
  "best_index": 0부터 {min(k_ui-1, len(candidates)-1)} 사이 정수,
  "best_topic": "선택한 토픽명",
  "reasoning": "선택 이유 (한국어 2-3문장)",
  "alternative_topics": ["다른 후보 토픽 (최대 3개)"]
}}
"""
            }
        ],
        response_format={"type": "json_object"},
        temperature=0
    )
    try:
        out = json.loads(response.choices[0].message.content)
    except Exception:
        out = {
            "confidence": "medium",
            "found": True,
            "best_index": 0,
            "best_topic": candidates[0][0].metadata.get("topic", ""),
            "reasoning": "판단 JSON 파싱에 실패하여 1순위 후보를 선택했습니다.",
            "alternative_topics": build_alternative_topics(candidates[1:], 3)
        }

    # safety
    bi = out.get("best_index", 0)
    if bi is None or not isinstance(bi, int) or bi < 0 or bi >= len(candidates):
        out["best_index"] = 0

    if out.get("confidence") == "low":
        out["found"] = False

    if not out.get("alternative_topics"):
        out["alternative_topics"] = build_alternative_topics(candidates[1:], 3)

    return out

def smart_search_v3(query: str, k_ui: int = 10, rewritten: Optional[str] = None, pinned_topic: Optional[str] = None) -> Dict[str, Any]:
    """
    Returns:
      dict with fields:
      - confidence, found, best_topic, reasoning, alternative_topics
      - rewritten, dist, rerank_score, doc, video_url, start_time, end_time
      - candidates (reranked)
    """
    if rewritten is None:
        rewritten = rewrite_query(query)

    # retrieve
    resolved_topic, raw = retrieve_candidates(query, rewritten, k_ui, pinned_topic=pinned_topic)

    if not raw:
        return {
            "found": False,
            "confidence": "low",
            "rewritten": rewritten,
            "message": "검색 결과가 없습니다.",
            "candidates": []
        }

    # raw: [(doc, dist), ...]
    # rerank: 후보 수는 k_ui*2 (최대 30) 수준에서만
    reranked = rerank(query, raw)

    # 2-stage로 토픽이 확정된 경우는 기본적으로 high로 처리 가능(특히 정의형 질문)
    if resolved_topic and is_basic_definition_question(query):
        best_doc, best_dist, best_rr = reranked[0]
        return {
            "confidence": "high",
            "found": True,
            "best_index": 0,
            "best_topic": resolved_topic,
            "reasoning": "질문이 기본 개념(정의/의미) 유형이며, 토픽이 DB에 존재해 해당 토픽 내부 구간에서 최적 후보를 선택했습니다.",
            "alternative_topics": build_alternative_topics(reranked[1:], 3),
            "rewritten": rewritten,
            "dist": best_dist,
            "rerank_score": best_rr,
            "doc": best_doc,
            "video_url": best_doc.metadata.get("video_url"),
            "start_time": safe_float(best_doc.metadata.get("start_time", 0)),
            "end_time": safe_float(best_doc.metadata.get("end_time", 0)),
            "candidates": reranked
        }

    # 그 외는 judge로 confidence/선택 index 결정
    judgment = judge_best(query, rewritten, reranked, k_ui)

    best_idx = judgment.get("best_index", 0)
    best_doc, best_dist, best_rr = reranked[best_idx]
    best_topic = judgment.get("best_topic") or best_doc.metadata.get("topic", "")

    return {
        **judgment,
        "rewritten": rewritten,
        "dist": best_dist,
        "rerank_score": best_rr,
        "doc": best_doc,
        "video_url": best_doc.metadata.get("video_url"),
        "start_time": safe_float(best_doc.metadata.get("start_time", 0)),
        "end_time": safe_float(best_doc.metadata.get("end_time", 0)),
        "best_topic": best_topic,
        "candidates": reranked
    }

# =========================
# UI
# =========================
st.title("📚 영어 학습 도우미 v3.1")
st.markdown("**개선**: Topic Routing + GPU Reranker (retrieve는 10~20개만, 그 안에서만 rerank)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.sidebar:
    st.header("⚙️ 설정")

    search_mode = st.radio(
        "검색 모드",
        ["🔍 하이브리드 (KG 우선)", "⚡ Smart Search만"],
        index=0,
        help="하이브리드: KG에서 먼저 찾고 없으면 Smart Search\nSmart Search만: 전체 토픽에서 직접 검색"
    )

    st.divider()

    k_value = st.slider(
        "표시/판단 후보 수 (k)",
        min_value=5,
        max_value=20,
        value=10,
        step=1,
        help="최종 후보 표시/판단에 쓰는 k. 내부 retrieve는 k*2 (최대 30)로만 확장됨."
    )

    st.divider()

    st.markdown("### 📚 KG 주제")
    st.caption(f"{len(knowledge_graph)}개")
    with st.expander("주제 목록 보기"):
        for topic in sorted(knowledge_graph.keys()):
            st.markdown(f"- {topic}")

    st.divider()

    st.markdown("### 📊 시스템")
    st.markdown(f"- Chroma topics: {len(ALL_TOPICS)}")
    st.markdown(f"- KG: {len(knowledge_graph)}")
    st.markdown(f"- Rewrite/Judge: {OPENAI_REWRITE_MODEL} / {OPENAI_JUDGE_MODEL}")
    st.markdown(f"- Explain: {OPENAI_EXPLAIN_MODEL}")
    st.markdown(f"- Reranker: {'ON' if RERANKER else 'OFF'} ({RERANKER_MODEL})")
    st.markdown(f"- E5 prefix: {'ON' if USE_E5_PREFIX else 'OFF'}")
    st.markdown("- 버전: v3.1")

    st.divider()
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()

user_input = st.chat_input("질문을 입력하세요...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):

            if "하이브리드" in search_mode:
                rewritten = rewrite_query(user_input)
                kg_result = search_in_knowledge_graph(rewritten)

                if kg_result:
                    st.caption("🔍 Knowledge Graph에서 찾음")
                    main_topic = kg_result["main_topic"]

                    st.markdown(f"## 💡 {main_topic}")
                    with st.spinner("설명 생성 중..."):
                        explain_prompt = f"'{main_topic}'이 무엇인지 핵심만 간단히 2-3문장으로 설명해주세요."
                        explanation = llm.invoke(explain_prompt).content.strip()
                    st.write(explanation)

                    st.markdown("---")
                    st.success(f"✅ '{main_topic}' 강의가 있어요!")

                    # 토픽 확정: pinned_topic으로 topic-filter 구간 검색
                    result = smart_search_v3(user_input, k_ui=k_value, rewritten=main_topic, pinned_topic=main_topic)

                else:
                    st.caption("⚡ KG에 없음 → Smart Search 사용")
                    result = smart_search_v3(user_input, k_ui=k_value, rewritten=rewritten)

            else:
                result = smart_search_v3(user_input, k_ui=k_value)

            # ===== Render =====
            keyword = result.get("best_topic") or user_input
            rewritten_q = result.get("rewritten", "")
            confidence = result.get("confidence", "low")
            dist = result.get("dist", 0.0)
            rr = result.get("rerank_score", 0.0)

            st.markdown(f"## 💡 {keyword}")
            st.caption(f"🔍 {user_input} → {rewritten_q} → {keyword}")

            if confidence == "high":
                st.success(f"✅ '{keyword}' 강의를 찾았어요!")
            elif confidence == "medium":
                st.info(f"🟡 '{keyword}' 토픽이 관련있을 수 있어요")
            else:
                st.warning(f"📭 '{keyword}'을(를) 직접 다루는 강의는 없어요")

            st.caption(f"🎯 확신도: {confidence} | dist: {dist:.4f} | rerank: {rr:.4f}")

            if result.get("reasoning"):
                with st.expander("💡 판단 근거"):
                    st.write(result["reasoning"])

            if confidence in ["high", "medium"]:
                video_url = result.get("video_url")
                start = safe_float(result.get("start_time", 0))
                end = safe_float(result.get("end_time", 0))

                if video_url:
                    st.markdown("### 📺 강의 영상")
                    embed_url = get_video_embed(video_url, start, end)
                    if embed_url:
                        st.components.v1.iframe(embed_url, width=800, height=450)
                        st.caption(f"⏱️ {int(start)}초 ~ {int(end)}초")

                st.markdown("---")
                st.markdown("### 🔗 관련 주제 (rerank 상위 후보)")
                shown = set()
                for i, (doc, d, rscore) in enumerate(result.get("candidates", [])[:6]):
                    t = doc.metadata.get("topic", "")
                    if not t or t in shown:
                        continue
                    shown.add(t)
                    with st.expander(f"✅ {t} (dist: {d:.3f}, rerank: {rscore:.3f})"):
                        rel_url = doc.metadata.get("video_url", "")
                        rel_start = safe_float(doc.metadata.get("start_time", 0))
                        rel_end = safe_float(doc.metadata.get("end_time", 0))
                        rel_embed = get_video_embed(rel_url, rel_start, rel_end)
                        if rel_embed:
                            st.components.v1.iframe(rel_embed, width=700, height=400)

            if confidence == "low" and result.get("alternative_topics"):
                st.markdown("### 🔍 이런 토픽들을 찾아보시겠어요?")
                for alt in result["alternative_topics"]:
                    st.markdown(f"- {alt}")
