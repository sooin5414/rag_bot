from __future__ import annotations

from typing import TypedDict, List, Optional, Literal, Dict, Any

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


# 1) State 정의 ---------------------------------------------------------------

class GraphState(TypedDict, total=False):
    messages: List[BaseMessage]

    question: str

    # 검색 후보들 (비디오 세그먼트 리스트)
    candidate_segments: List[Dict[str, Any]]
    # 선택된 세그먼트
    selected_segment: Optional[Dict[str, Any]]

    answer: Optional[str]
    score: Optional[Literal["GOOD", "BAD"]]


# 2) Helper: question 추출 ---------------------------------------------------

def prepare_question_node(state: GraphState) -> Dict:
    msgs = state.get("messages", [])
    q = state.get("question")

    if not q:
        for m in reversed(msgs):
            if isinstance(m, HumanMessage):
                q = m.content
                break

    if q is None:
        q = ""

    return {"question": q}


# 3) TODO: 네 비디오 인덱스/LLM hook ---------------------------------------

def video_retriever_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    # TODO: 여기서 rag_bot 비디오 인덱스 호출
    return [
        {
            "video_id": "demo_video_id",
            "start": 10.0,
            "end": 25.0,
            "text": "TODO: 실제 자막 세그먼트 텍스트",
            "score": 0.99,
        }
    ]


def call_llm_answer(context: str, question: str) -> str:
    # TODO: OpenAI GPT-4o 등으로 교체
    return f"[DEMO] context 기반 답변: {question[:20]}..."


def call_llm_judge(question: str, context: str, answer: str) -> Literal["GOOD", "BAD"]:
    # TODO: 나중에 LLM-as-judge로 교체
    return "GOOD"


# 4) 각 노드 구현 ------------------------------------------------------------

def retrieve_segments_node(state: GraphState) -> Dict:
    q = state.get("question", "")
    segments = video_retriever_search(q, top_k=5)
    return {"candidate_segments": segments}


def select_segment_node(state: GraphState) -> Dict:
    candidates = state.get("candidate_segments") or []
    selected = candidates[0] if candidates else None
    return {"selected_segment": selected}


def generate_answer_node(state: GraphState) -> Dict:
    msgs = state.get("messages", [])
    q = state.get("question", "")
    seg = state.get("selected_segment")

    if not seg:
        ai = AIMessage(content="관련된 영상 세그먼트를 찾지 못했습니다.")
        return {"answer": ai.content, "messages": msgs + [ai]}

    context = seg["text"]
    answer = call_llm_answer(context=context, question=q)

    ai = AIMessage(
        content=answer,
        additional_kwargs={
            "video_segment": {
                "video_id": seg["video_id"],
                "start": seg["start"],
                "end": seg["end"],
                "score": seg.get("score"),
            }
        },
    )

    return {"answer": answer, "messages": msgs + [ai]}


def evaluate_answer_node(state: GraphState) -> Dict:
    q = state.get("question", "")
    seg = state.get("selected_segment")
    answer = state.get("answer") or ""
    context = seg["text"] if seg else ""
    score = call_llm_judge(q, context, answer)
    return {"score": score}


# 5) 조건 분기 ---------------------------------------------------------------

def route_on_score(state: GraphState) -> str:
    if state.get("score") == "BAD":
        return "retry"
    return "end"


# 6) 그래프 구성 -------------------------------------------------------------

def build_graph():
    g = StateGraph(GraphState)

    g.add_node("prepare_question", prepare_question_node)
    g.add_node("retrieve_segments", retrieve_segments_node)
    g.add_node("select_segment", select_segment_node)
    g.add_node("generate_answer", generate_answer_node)
    g.add_node("evaluate_answer", evaluate_answer_node)

    g.set_entry_point("prepare_question")

    g.add_edge("prepare_question", "retrieve_segments")
    g.add_edge("retrieve_segments", "select_segment")
    g.add_edge("select_segment", "generate_answer")
    g.add_edge("generate_answer", "evaluate_answer")

    g.add_conditional_edges(
        "evaluate_answer",
        route_on_score,
        {
            "retry": "retrieve_segments",
            "end": END,
        },
    )

    app = g.compile(checkpointer=MemorySaver())
    return app


graph = build_graph()


if __name__ == "__main__":
    internal = graph.get_graph()
    print(internal.draw_mermaid())
    internal.draw_mermaid_png("video_rag_detailed_pipeline.png")
