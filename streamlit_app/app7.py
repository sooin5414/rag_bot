#!/data/edutem/.cache/pypoetry/virtualenvs/rag-bot-vbdTYmCJ-py3.12/bin/python
"""
카드 형식 UI + 다중 답변 지원
"""
import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import torch
from openai import OpenAI
import json
import re

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

vectorstore = load_vectorstore()
reranker = load_reranker()

# =====================
# Core Functions
# =====================
def analyze_query_intent(query: str) -> dict:
    """질문 의도 파악 (LLM)"""
    prompt = f"""
    아래 질문을 분석해서 JSON으로 반환하세요.

    질문: {query}

    반환 형식:
    {{
        "grammar_type": "시제 | be동사 | 조동사 | 문장구조 | 전치사 | 동사 | 기타",
        "specificity": "broad | specific",
        "desired_role": "definition | usage | comparison | practice"
    }}

    판단 기준:
    - broad: "현재시제가 뭐야?", "be동사 알려줘" 같은 포괄적 질문
    - specific: "현재완료 예문 보여줘", "can의 용법" 같은 구체적 질문
    - desired_role: 질문에서 원하는 내용 (뭐야?→definition, 어떻게?→usage, 예문→practice)
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(response.choices[0].message.content)


def smart_search(query: str) -> dict:
    """개선된 검색: 신뢰도 + 의도 기반"""
    # Step 1: 의도 파악
    intent = analyze_query_intent(query)

    # Step 2: Vector Search + Rerank
    docs = vectorstore.similarity_search(query, k=20)
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked_indices = scores.argsort()[::-1]
    ranked_docs = [docs[i] for i in ranked_indices]
    ranked_scores = [scores[i] for i in ranked_indices]

    top_score = ranked_scores[0]

    # Step 3: 신뢰도 + 의도 조합 판단
    if top_score > 0.7:
        # 고신뢰도 → 단일 답변
        return {
            "mode": "single",
            "doc": ranked_docs[0],
            "score": float(top_score),
            "confidence": "high",
            "intent": intent
        }

    elif intent['specificity'] == 'broad' and top_score < 0.7:
        # 포괄적 질문 + 저신뢰도 → 여러 개 보여주기
        filtered = [
            (d, s) for d, s in zip(ranked_docs[:10], ranked_scores[:10])
            if d.metadata.get('grammar_type') == intent['grammar_type']
        ]

        # Role 우선순위 정렬
        role_priority = {'definition': 3, 'usage': 2, 'comparison': 1, 'practice': 1}
        filtered.sort(
            key=lambda x: (
                role_priority.get(x[0].metadata.get('role'), 0),
                x[1]  # score
            ),
            reverse=True
        )

        top_docs = [d for d, s in filtered[:3]]
        top_scores = [s for d, s in filtered[:3]]

        return {
            "mode": "multi",
            "docs": top_docs,
            "scores": [float(s) for s in top_scores],
            "message": f"'{intent['grammar_type']}' 관련 주제들입니다:",
            "confidence": "medium",
            "intent": intent
        }

    else:
        # 구체적 질문 + 중신뢰도 → Top 3 선택지
        return {
            "mode": "choice",
            "docs": ranked_docs[:3],
            "scores": [float(s) for s in ranked_scores[:3]],
            "message": "혹시 이 중 하나를 찾으시나요?",
            "confidence": "medium",
            "intent": intent
        }


def youtube_embed(video_url: str, start: float, end: float) -> str:
    """YouTube embed URL 생성"""
    match = re.search(r'(?:v=|/)([a-zA-Z0-9_-]{11})', video_url)
    if not match:
        return None
    vid = match.group(1)
    return f"https://www.youtube.com/embed/{vid}?start={int(start)}&end={int(end)}"


# =====================
# UI Styling
# =====================
st.set_page_config(page_title="영어 학습 도우미")

# Custom CSS for chat-like UI
st.markdown("""
<style>
    .user-message {
        background-color: #E3F2FD;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }

    .bot-message {
        background-color: #F5F5F5;
        padding: 15px 20px;
        border-radius: 15px;
        margin: 10px 0;
        margin-right: 20%;
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

    /* 입력창 스타일링 */
    [data-testid="stChatInput"] {
        max-width: 800px;
        margin: 0 auto;
    }

    /* 입력창을 포함하는 컨테이너 */
    [data-testid="stBottom"] {
        background-color: white;
        border-top: 1px solid #e0e0e0;
        padding: 1rem 0;
    }

    /* 메시지 영역에 하단 여백 추가 */
    .main .block-container {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Main UI
# =====================
st.title("🎓 영어 학습 도우미")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        # Display videos if present
        if "videos" in msg:
            for video in msg["videos"]:
                with st.expander(f"📺 {video['topic']}", expanded=True):
                    col1, col2 = st.columns([3, 2])

                    with col1:
                        embed_url = video.get('embed_url')
                        if embed_url:
                            st.components.v1.iframe(embed_url, width=500, height=300)

                    with col2:
                        st.markdown(f"**{video['topic']}**")
                        st.caption(f"🏷️ {video['grammar_type']} | {video['role']}")
                        st.write(video['content'])
                        st.caption(f"⏱️ {video['start']:.1f}s ~ {video['end']:.1f}s")

# Input
query = st.chat_input("질문을 입력하세요 (예: 현재시제가 뭐야?)")

if query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Search
    with st.spinner("검색 중..."):
        result = smart_search(query)

    # Generate response based on mode
    if result["mode"] == "single":
        doc = result["doc"]
        meta = doc.metadata

        response_text = f"**{meta.get('topic')}**\n\n{doc.page_content}"

        video_info = [{
            "topic": meta.get('topic'),
            "grammar_type": meta.get('grammar_type'),
            "role": meta.get('role'),
            "content": doc.page_content,
            "start": meta.get('start_time', 0),
            "end": meta.get('end_time', 0),
            "embed_url": youtube_embed(meta.get('video_url', ''), meta.get('start_time', 0), meta.get('end_time', 0))
        }]

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "videos": video_info
        })

    elif result["mode"] in ["multi", "choice"]:
        response_text = result["message"]

        video_info = []
        for doc in result["docs"]:
            meta = doc.metadata
            video_info.append({
                "topic": meta.get('topic'),
                "grammar_type": meta.get('grammar_type'),
                "role": meta.get('role'),
                "content": doc.page_content,
                "start": meta.get('start_time', 0),
                "end": meta.get('end_time', 0),
                "embed_url": youtube_embed(meta.get('video_url', ''), meta.get('start_time', 0), meta.get('end_time', 0))
            })

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "videos": video_info
        })

    st.rerun()
