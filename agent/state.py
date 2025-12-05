from typing import TypedDict, Literal, List, Optional, Dict, Any
from langchain_core.messages import BaseMessage


class GraphState(TypedDict, total=False):
    # 1) 대화 전체 메시지 히스토리
    #    agent-chat-ui / LangGraph 기본 패턴
    messages: List[BaseMessage]

    # 2) 질의
    #    - prepare_question 노드에서 messages에서 뽑아서 채움
    question: str

    # 3) 비디오 RAG용 retrieval 결과들
    #    - retrieve_segments 노드에서 채움
    #    - 각 세그먼트 예시:
    #      {"video_id": "...", "start": 10.0, "end": 25.0, "text": "...", "score": 0.87}
    candidate_segments: List[Dict[str, Any]]

    # 4) 실제로 선택된 세그먼트 (자동 재생 대상)
    #    - select_segment 노드에서 채움
    selected_segment: Optional[Dict[str, Any]]

    # 5) LLM 답변 텍스트
    #    - generate_answer 노드에서 채움
    answer: Optional[str]

    # 6) 품질 평가 결과 (Adaptive RAG에서 사용)
    #    - evaluate_answer 노드에서 채움
    score: Optional[Literal["GOOD", "BAD"]]