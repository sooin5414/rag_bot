import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import streamlit.components.v1 as components
import os


# ============================================================
# 1. 페이지 설정
# ============================================================

st.set_page_config(
    page_title="영어 학습 도우미",
    page_icon="📚",
)

st.title("📚 영어 학습 도우미")
st.markdown("영상 검색 시스템")

# ============================================================
# 2. 벡터스토어 로드
# ============================================================

@st.cache_resource
def load_vectorstore():    
    with st.spinner("벡터스토어 로딩 중..."):
        if os.path.exists("./lecture_db/chroma.sqlite3"):
            embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-large"
        )
        
        # 이미 만들어진 벡터스토어 로드
            vectorstore = Chroma(
            persist_directory="./lecture_db",
            embedding_function=embeddings,
            collection_name="lectures_v1"
            )
            #st.write(f"벡터스토어 문서 개수: {vectorstore._collection.count()}")
    
    return vectorstore

vectorstore = load_vectorstore()

# ============================================================
# 3. 세션 상태
# ============================================================

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# ============================================================
# 4. 사이드바
# ============================================================

with st.sidebar:
    st.header("⚙️ 설정")
    
    # 검색 결과 수
    num_results = st.slider("검색 결과 수", 1, 5, 3)
    
    st.divider()
    
    # 초기화 버튼
    if st.button("🗑️ 대화 초기화"):
        st.session_state["messages"] = []
        st.rerun()
    
    st.divider()

# ============================================================
# 5. 메인 영역
# ============================================================

# 이전 대화 출력
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# 사용자 입력
user_input = st.chat_input("질문을 입력하세요... (예: at은 언제 써?)")

if user_input:
    # 사용자 메시지 추가
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input
    })
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # AI 응답
    with st.chat_message("assistant"):
        with st.spinner("검색 중..."):
            
            # 검색
            results = vectorstore.similarity_search(
                user_input, 
                k=num_results
            )
            
             # 검색된 결과 확인 (디버깅 추가)
            if len(results) == 0:
                st.error("검색 결과가 없습니다. 다른 질문을 시도해 주세요.")
            
            # 결과 출력
            st.markdown("📺 **검색 결과:**")
            st.markdown("")
            
            response_text = "📺 **검색 결과:**\n\n"
            
            for i, doc in enumerate(results, 1):
                url = doc.metadata['video_url']
                start = int(doc.metadata['start_time'])
                content = doc.page_content
                
                with st.container():
                    st.markdown("-------------------")
                    
                    video_id = url.split("youtu.be/")[-1].split("?")[0]
                    embed_url = f"https://www.youtube.com/embed/{video_id}?start={start}"
                    
                    st.components.v1.iframe(embed_url, width = 800, height=500)
                    
                    # 영상 링크 버튼
                    st.link_button(
                        f"▶️ YouTube에서 보기 ({start}초)",
                        f"{url}&t={start}s",
                              )
                    
                    st.markdown("---")
                
                # 응답 텍스트에 추가
                response_text += f"[{i}] {content[:100]}...\n"
                response_text += f"🎬 {url}&t={start}s\n\n"
            
            # 메시지 저장
            st.session_state["messages"].append({
                "role": "assistant",
                "content": response_text
            })