import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import torch
from openai import OpenAI
import json

load_dotenv()
client = OpenAI()


# =====================
# Config
# =====================
CHROMA_DIR = "/data/edutem/sooine/rag_bot/chroma_db_with_role"
EMBED_MODEL = "intfloat/multilingual-e5-large"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# =====================
# Load resources
# =====================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

@st.cache_resource
def load_reranker():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(RERANK_MODEL, device=device, max_length=512)

@st.cache_resource
def load_all_topics(_vectorstore):
    metas = _vectorstore._collection.get(include=["metadatas"])["metadatas"]
    return list({m["topic"] for m in metas if m and "topic" in m})

vectorstore = load_vectorstore()
reranker = load_reranker()

ALL_TOPICS = load_all_topics(vectorstore)

def extract_asked_topic(query: str, all_topics: list[str]) -> str | None:
    q = query.replace(" ", "")
    for t in all_topics:
        if t.replace(" ", "") in q:
            return t
    return None

def rewrite_query_structured(query: str, all_topics: list[str]) -> dict:
    topics = all_topics[:180]

    prompt = f"""
            너는 영어 문법 교육 시스템의 Query Router다.

            사용자 질문을 보고,
            1.  가장 적절한 영어 문법 '토픽 1개'
            2. 사용자의 핵심 의도(role) 1개만
            을 추출하라.

            role 정의:
            - definition : 개념/의미가 무엇인지
            - usage      : 언제/어떻게 쓰는지
            - comparison : 다른 문법과의 차이
            - practice   : 예문/연습/문제

            중요 규칙:
            - role은 반드시 1개만 선택
            - topic이 정확히 없으면, 가장 가까운 상위 개념을 선택
            - 구어체, 오타, 붙여쓴 표현은 정상화해서 판단
            - 확신이 낮으면 confidence를 낮게 설정

            오타 처리 예시:
            - "투부정사" → "to부정사"
            - "비동사" → "be동사"
            - "현제완료" → "현재완료"
            - "과거형" → "과거"

            사용자 질문:
            {query}

            토픽 후보:
            {topics}

            출력(JSON only):
            {{
            "topic": "선택한 토픽 (반드시 1개)",
            "role": "definition | usage | comparison | practice",
            "confidence": "high | medium | low"
            }}
"""

    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=80
    )

    return json.loads(res.choices[0].message.content)

def has_definition(topic: str, docs) -> bool:
    for d in docs:
        if (
            d.metadata.get("topic") == topic and
            d.metadata.get("role") == "definition"
        ):
            return True
    return False

def decide_educational_response(query, routed, retrieved_docs):
    topic  = routed["topic"]
    intent = routed["intent"]

    # 1. 개념 자체가 없음
    if topic not in ALL_TOPICS:
        return {
            "type": "concept_missing",
            "topic": topic
        }

    # 2. 정의 질문인데 정의 강의 없음
    if intent == "definition" and not has_definition(topic, retrieved_docs):
        return {
            "type": "definition_missing",
            "topic": topic
        }

    # 3. 정상 → 영상 선택 가능
    return {
        "type": "video_ok",
        "topic": topic
    }

def build_definition_missing_message(topic: str, vectorstore):
    # 같은 grammar_type의 다른 topic 수집
    metas = vectorstore._collection.get(include=["metadatas"])["metadatas"]

    related = {}
    for m in metas:
        if not m:
            continue
        if m.get("grammar_type") == "시제" and m.get("topic") != topic:
            related[m["topic"]] = m.get("summary", "")

    lines = [f"- **{t}**: {s}" for t, s in related.items()]

    return f"""
⚠️ **{topic}** 자체를 설명하는 강의는 아직 없습니다.

대신, 현재와 관련된 시제 강의는 다음이 있습니다:
{chr(10).join(lines)}

👉 **{topic} (습관 / 사실)** 에 대한 강의는 준비 중입니다.
"""
def generate_concept_summary(query: str) -> str:
    prompt = f"""
너는 영어 문법을 설명하는 전문 튜터다.

사용자의 질문에 대해:
- 질문에서 묻는 개념 자체를 설명하라
- 2~3문장 이내로 간결하게 설명하라
- 강의, 영상, 자료의 존재 여부는 절대 언급하지 마라
- 질문이 모호하면, 가장 일반적인 문법적 의미를 기준으로 설명하라
- 다른 문법 개념으로 바꿔서 설명하지 마라

사용자 질문:
{query}

출력:
- 한국어 문장만 출력
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=120,
        )
        return res.choices[0].message.content.strip()
    except Exception:
        return ""

# =====================
# Core logic
# =====================
def search_video(query: str, k: int = 12):
    # =========================
    # 0. Routing
    # =========================
    routed = rewrite_query_structured(query, ALL_TOPICS)

    asked_topic = extract_asked_topic(query, ALL_TOPICS)      # 사용자가 실제로 물은 개념
    candidate_topic = routed.get("topic")                     # 검색용 토픽
    role = routed.get("role", "definition")                   # definition / usage / ...
    
    # =========================
    # 🔴 핵심 가드: asked_topic이 DB에 없으면 검색 금지
    # =========================
    if role == "definition" and asked_topic:
        if asked_topic not in ALL_TOPICS:
            return {
                "type": "definition_missing",
                "topic": asked_topic,
                "related_topics": [],
                "routed": routed,
            }
    
    
    route_conf = routed.get("confidence", "low")

    # =========================
    # 1. Vector search (candidate_topic 기준)
    # =========================
    if isinstance(candidate_topic, str) and candidate_topic.strip():
        raw = vectorstore.similarity_search_with_score(
            query,
            k=k,
            filter={"topic": candidate_topic}
        )
    else:
        raw = vectorstore.similarity_search_with_score(query, k=k)

    if not raw:
        return None

    retrieved_docs = [doc for doc, _ in raw]

    # =========================
    # 2. Definition-missing 판단 (asked_topic 기준)
    # =========================
    if role == "definition" and asked_topic:
        if not has_definition(asked_topic, retrieved_docs):

            # 같은 grammar_type의 다른 토픽 수집
            metas = vectorstore._collection.get(include=["metadatas"])["metadatas"]

            target_grammar = None
            for d in retrieved_docs:
                if d.metadata.get("topic") == candidate_topic:
                    target_grammar = d.metadata.get("grammar_type")
                    break

            related_topics = []
            seen = set()
            for m in metas:
                if not m:
                    continue
                t = m.get("topic")
                if (
                    m.get("grammar_type") == target_grammar
                    and t != asked_topic
                    and t not in seen
                ):
                    related_topics.append((t, m.get("summary", "")))
                    seen.add(t)

            return {
                "type": "definition_missing",
                "topic": asked_topic,
                "related_topics": related_topics[:5],
                "routed": routed,
            }

    # =========================
    # 3. Rerank
    # =========================
    pairs = [(query, doc.page_content) for doc, _ in raw]
    rr_scores = reranker.predict(pairs)

    scored = []
    for (doc, dist), rr in zip(raw, rr_scores):
        final = rr - 0.2 * dist

        doc_role = doc.metadata.get("role")
        if doc_role == role:
            final *= 1.25
        elif role == "definition" and doc_role != "definition":
            final *= 0.95

        scored.append((doc, dist, rr, final))

    scored.sort(key=lambda x: x[3], reverse=True)
    best_doc, best_dist, best_rr, best_final = scored[0]

    # =========================
    # 4. Confidence 계산
    # =========================
    if len(scored) > 1:
        margin = scored[0][3] - scored[1][3]
    else:
        margin = 0.15

    if route_conf == "high":
        m_hi, m_mid = 0.2, 0.08
    else:
        m_hi, m_mid = 0.3, 0.12

    if margin > m_hi:
        confidence = "high"
    elif margin > m_mid:
        confidence = "medium"
    else:
        confidence = "low"

    # =========================
    # 5. 정상 영상 반환
    # =========================
    return {
        "type": "video_ok",
        "doc": best_doc,
        "confidence": confidence,
        "final_score": best_final,
        "rerank": best_rr,
        "dist": best_dist,
        "routed": routed,
    }


def youtube_embed(url, start, end):
    if "watch?v=" in url:
        vid = url.split("watch?v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        vid = url.split("youtu.be/")[1].split("?")[0]
    else:
        return None
    return f"https://www.youtube.com/embed/{vid}?start={int(start)}&end={int(end)}"

# =====================
# UI
# =====================
st.set_page_config(page_title="영어 학습 도우미")

# Custom CSS for chat-like UI
st.markdown("""
<style>
    .message-row {
        display: flex !important;
        margin: 10px 0 !important;
        clear: both !important;
        width: 100% !important;
    }

    .message-row.user {
        justify-content: flex-start !important;
        flex-direction: row-reverse !important;
    }

    .message-row.bot {
        justify-content: flex-start !important;
    }

    .avatar {
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        border-radius: 50% !important;
        background-color: #E0E0E0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-size: 20px !important;
        margin: 0 10px !important;
    }

    .user-message {
        background-color: #E3F2FD !important;
        padding: 15px 20px !important;
        border-radius: 15px !important;
        max-width: 70% !important;
        width: fit-content !important;
        text-align: right !important;
    }

    .bot-message {
        background-color: #F5F5F5 !important;
        padding: 15px 20px !important;
        border-radius: 15px !important;
        max-width: 80% !important;
        width: fit-content !important;
    }

    .video-card {
        background-color: white;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    .topic-badge {
        display: inline-block;
        background-color: #2196F3;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
        margin-right: 10px;
    }

    .role-badge {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.85em;
    }

    /* 입력창 원래대로 */
    /* 메시지 영역에 하단 여백 추가 */
    .main .block-container {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)
import time
from concurrent.futures import ThreadPoolExecutor
def stream_text(placeholder, text):
    """스트리밍 효과로 텍스트 출력"""
    displayed = ""
    for char in text:
        displayed += char
        placeholder.markdown(displayed)
        time.sleep(0.01)
        
st.title("🎓 영어학습도우미")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (이전 대화만)
if len(st.session_state.messages) > 0:
    st.markdown("### 💬 대화 기록")
    for msg in st.session_state.messages:
        if msg.get("role") == "user":
            st.markdown(
                f'''<div class="message-row user">
                    <div class="avatar">👤</div>
                    <div class="user-message">{msg.get("content", "")}</div>
                </div>''',
                unsafe_allow_html=True
            )
        elif msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:  # Only display if there's actual content
                st.markdown(
                    f'''<div class="message-row bot">
                        <div class="avatar">💡</div>
                        <div class="bot-message">{content}</div>
                    </div>''',
                    unsafe_allow_html=True
                )
    st.markdown("---")  # 구분선

query = st.chat_input("질문을 입력하세요 (예: 비동사가 뭐야?)")

if query:
    # 1. 사용자 메시지 추가 (세션에 저장)
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # 현재 질문 말풍선으로 표시
    st.markdown(
        f'''<div class="message-row user">
            <div class="avatar">👤</div>
            <div class="user-message">{query}</div>
        </div>''',
        unsafe_allow_html=True
    )

    # ✅ 병렬 시작: 요약 생성 & 영상 검색 동시에 백그라운드 실행
    with ThreadPoolExecutor(max_workers=2) as executor:
        summary_future = executor.submit(generate_concept_summary, query)
        video_future = executor.submit(search_video, query)

        # 요약 결과 받기
        summary = summary_future.result()

        # 2. 요약 스트리밍 (말풍선 스타일로)
        summary_placeholder = st.empty()

        displayed = ""
        for char in summary:
            displayed += char
            # 말풍선 스타일로 스트리밍
            summary_html = f'''<div class="message-row bot">
                <div class="avatar">🤖</div>
                <div class="bot-message">{displayed}</div>
            </div>'''
            summary_placeholder.markdown(summary_html, unsafe_allow_html=True)
            time.sleep(0.01)

        # 요약 메시지 저장
        st.session_state.messages.append({
            "role": "assistant",
            "content": summary,
            "type": "summary"
        })
        
        # 영상 검색이 아직 안 끝났으면 대기
        with st.spinner("🔄 영상 로딩 중..."):
            result = video_future.result()
    
    # 4. 영상 결과 처리
    if not result:
        st.error("관련 영상을 찾지 못했습니다.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "관련 영상을 찾지 못했습니다.",
            "type": "text"
        })
    
    elif result.get("type") == "definition_missing":
        related_text = "\n".join([
            f"- **{t}**: {desc}" 
            for t, desc in result["related_topics"]
        ])
        bot_msg = f"""⚠️ **{result['topic']}** 자체를 설명하는 강의는 아직 없습니다.

### 📚 대신, 현재 제공되는 관련 강의
{related_text}

👉 **{result['topic']} (기본 개념)** 강의는 준비 중입니다."""
        
        st.warning(bot_msg)
        st.session_state.messages.append({
            "role": "assistant",
            "content": bot_msg,
            "type": "text"
        })
    
    elif result.get("type") == "video_ok":
        doc = result["doc"]
        meta = doc.metadata
        topic = meta.get("topic", "Unknown")

        # 답변을 말풍선으로 표시
        answer_html = f'''<div class="message-row bot">
            <div class="avatar">🤖</div>
            <div class="bot-message"><strong>{topic}</strong></div>
        </div>'''
        st.markdown(answer_html, unsafe_allow_html=True)

        if result['confidence'] == "low":
            st.warning("⚠️ 정확도가 다소 낮을 수 있는 추천입니다.")

        embed_url = youtube_embed(
            meta.get("video_url", ""),
            meta.get("start_time", 0),
            meta.get("end_time", 0)
        )
        if embed_url:
            st.components.v1.iframe(embed_url, width=700, height=400)
        
        st.session_state.messages.append({
            "role": "assistant",
            "type": "video",
            "confidence": result['confidence'],
            "video_data": {
                "topic": topic,
                "url": meta.get("video_url", ""),
                "start": meta.get("start_time", 0),
                "end": meta.get("end_time", 0)
            }
        })