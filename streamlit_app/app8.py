#!/data/edutem/.cache/pypoetry/virtualenvs/rag-bot-vbdTYmCJ-py3.12/bin/python
"""
간단한 채팅 UI - 질문과 답변만 말풍선으로 표시
"""
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

# =====================
# UI Styling
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

    /* 메시지 영역에 하단 여백 추가 */
    .main .block-container {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# =====================
# Main UI
# =====================
st.title("🎓 영어 학습 도우미 - test ")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
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
            st.markdown(
                f'''<div class="message-row bot">
                    <div class="avatar">🤖</div>
                    <div class="bot-message">{msg.get("content", "")}</div>
                </div>''',
                unsafe_allow_html=True
            )
    st.markdown("---")

# Input
query = st.chat_input("질문을 입력하세요 (예: be 동사가 뭐야?)")

if query:
    # 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    # 현재 질문 표시
    st.markdown(
        f'''<div class="message-row user">
            <div class="avatar">👤</div>
            <div class="user-message">{query}</div>
        </div>''',
        unsafe_allow_html=True
    )

    # LLM 답변 생성
    with st.spinner("답변 생성 중..."):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 영어 문법을 설명하는 친절한 선생님이다. 간결하고 명확하게 설명하라."},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=500
        )
        answer = response.choices[0].message.content

    # 답변 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    # 답변 표시
    st.markdown(
        f'''<div class="message-row bot">
            <div class="avatar">🤖</div>
            <div class="bot-message">{answer}</div>
        </div>''',
        unsafe_allow_html=True
    )
