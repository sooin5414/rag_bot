"""
RAG 시스템 자동 평가 스크립트
- ChromaDB의 113개 토픽 기준으로 테스트 질문 생성
- 각 질문에 대해 올바른 토픽/영상이 검색되는지 평가
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import json
from collections import defaultdict

load_dotenv()

# 벡터스토어 로드
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(
    persist_directory="/data/edutem/sooine/rag_bot/chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 지식 그래프 로드
with open('knowledge_graph.json', 'r', encoding='utf-8') as f:
    knowledge_graph = json.load(f)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# rewrite_query 함수 (앱과 동일)
def rewrite_query(query):
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

def search_in_knowledge_graph(query):
    """앱과 동일한 지식 그래프 검색"""
    query_lower = query.lower().strip()
    topic_list = list(knowledge_graph.keys())

    # 1단계: 정확한 일치
    for main_topic in topic_list:
        if main_topic.lower() == query_lower:
            return main_topic

    # 2단계: 부분 일치
    for main_topic in topic_list:
        if query_lower in main_topic.lower() or main_topic.lower() in query_lower:
            return main_topic

    return None

# ChromaDB에서 모든 토픽 가져오기
collection = vectorstore._collection
results = collection.get(include=['metadatas'])

topics_dict = defaultdict(list)
for metadata in results['metadatas']:
    topic = metadata.get('topic', '')
    if topic:
        topics_dict[topic].append({
            'video_url': metadata.get('video_url', ''),
            'start_time': metadata.get('start_time', 0)
        })

# 전체 113개 토픽 사용 (113개 × 5 = 565개 질문)
main_topics = sorted(topics_dict.keys())

# 테스트 질문 생성 - 각 토픽당 5개씩
test_cases = []
question_templates = [
    "{}가 뭐야?",
    "{} 설명해줘",
    "{} 어떻게 써?",
    "{}에 대해 알려줘",
    "{}를 사용하는 방법",
]

for topic in main_topics:
    expected_videos = [item['video_url'] for item in topics_dict[topic]]

    # 각 토픽당 5개 질문 모두 생성
    for template in question_templates:
        question = template.format(topic)
        test_cases.append({
            'question': question,
            'expected_topic': topic,
            'expected_videos': expected_videos
        })

print(f"총 {len(test_cases)}개 테스트 케이스 생성")

# 평가 시작
results = []
correct_topic = 0
correct_video = 0

print("\n" + "="*80)
print("평가 시작")
print("="*80)

for i, test in enumerate(test_cases, 1):  # 전체 테스트
    question = test['question']
    expected_topic = test['expected_topic']
    expected_videos = test['expected_videos']

    # 순수 RAG만 사용 (지식 그래프 없이)
    rag_docs = retriever.invoke(question)

    if rag_docs:
        actual_topic = rag_docs[0].metadata.get('topic', '')
        actual_video = rag_docs[0].metadata.get('video_url', '')
        actual_start = rag_docs[0].metadata.get('start_time', 0)

        # 토픽 일치 여부 (정확히 또는 부분 일치)
        topic_match = (expected_topic == actual_topic) or \
                      (expected_topic in actual_topic) or \
                      (actual_topic in expected_topic)

        # 영상 일치 여부
        video_match = actual_video in expected_videos

        if topic_match:
            correct_topic += 1
        if video_match:
            correct_video += 1

        status = "✅" if (topic_match and video_match) else "❌"

        print(f"\n{i}. {status} {question}")
        print(f"   기대 토픽: {expected_topic}")
        print(f"   실제 토픽: {actual_topic}")
        if not video_match:
            print(f"   ⚠️ 영상 불일치!")

        results.append({
            'question': question,
            'expected_topic': expected_topic,
            'actual_topic': actual_topic,
            'topic_match': topic_match,
            'video_match': video_match,
            'start_time': actual_start
        })
    else:
        print(f"\n{i}. ❌ {question}")
        print(f"   검색 결과 없음!")
        results.append({
            'question': question,
            'expected_topic': expected_topic,
            'actual_topic': None,
            'topic_match': False,
            'video_match': False
        })

# 평가 결과
print("\n" + "="*80)
print("평가 결과")
print("="*80)
print(f"토픽 정확도: {correct_topic}/{len(results)} ({correct_topic/len(results)*100:.1f}%)")
print(f"영상 정확도: {correct_video}/{len(results)} ({correct_video/len(results)*100:.1f}%)")

# 결과 저장
with open('evaluation_results.json', 'w', encoding='utf-8') as f:
    json.dump({
        'total_cases': len(results),
        'topic_accuracy': correct_topic / len(results),
        'video_accuracy': correct_video / len(results),
        'details': results
    }, f, ensure_ascii=False, indent=2)

print("\n평가 결과가 evaluation_results.json에 저장되었습니다.")
