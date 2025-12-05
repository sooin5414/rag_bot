"""
RAG 평가 스크립트
- Retrieval 품질 측정
- Answer 품질 측정 (RAGAS)
"""
import json
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

# =============================================================================
# 1. 테스트셋 정의
# =============================================================================

TEST_SET = [
    {
        "question": "be동사가 뭐야?",
        "expected_keywords": ["be동사", "am", "is", "are"],
        "expected_topic": "be동사",
    },
    {
        "question": "at, on, in 차이가 뭐야?",
        "expected_keywords": ["at", "on", "in", "전치사", "시간", "장소"],
        "expected_topic": "전치사",
    },
    {
        "question": "I am busy 맞는 문장이야?",
        "expected_keywords": ["am", "busy", "형용사"],
        "expected_topic": "be동사",
    },
    {
        "question": "현재완료 언제 써?",
        "expected_keywords": ["have", "has", "과거분사", "현재완료"],
        "expected_topic": "현재완료",
    },
    {
        "question": "수동태 만드는 법",
        "expected_keywords": ["be", "과거분사", "수동태"],
        "expected_topic": "수동태",
    },
]


# =============================================================================
# 2. Vectorstore 로드
# =============================================================================

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="english_grammar_chunked",
    )
    return vectorstore


# =============================================================================
# 3. 제외할 topic 목록
# =============================================================================

EXCLUDE_TOPICS = [
    "강의 마무리", "강의 종료", "수업 종료", "인사",
    "강의 시작", "수업 시작", "마무리"
]


# =============================================================================
# 4. Retrieval 평가
# =============================================================================

def evaluate_retrieval(vectorstore, test_set, k=5, use_filter=True):
    """
    각 질문에 대해 검색 결과 평가
    - Hit Rate: 관련 키워드가 top-k에 포함되는지
    - MRR: 첫 번째 관련 결과의 순위
    """
    results = []

    for item in test_set:
        query = item["question"]
        expected_keywords = item["expected_keywords"]

        # 검색 실행 (필터 적용 여부)
        if use_filter:
            docs_with_scores = vectorstore.similarity_search_with_score(
                query,
                k=k,
                filter={"topic": {"$nin": EXCLUDE_TOPICS}}
            )
        else:
            docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

        # 결과 분석
        hit = False
        first_hit_rank = None
        retrieved_texts = []

        for rank, (doc, score) in enumerate(docs_with_scores, 1):
            text = doc.page_content.lower()
            retrieved_texts.append({
                "rank": rank,
                "score": float(score),
                "text": doc.page_content[:200],
                "metadata": doc.metadata,
            })

            # 키워드 매칭 확인
            for kw in expected_keywords:
                if kw.lower() in text:
                    hit = True
                    if first_hit_rank is None:
                        first_hit_rank = rank
                    break

        results.append({
            "question": query,
            "expected_topic": item["expected_topic"],
            "hit": hit,
            "first_hit_rank": first_hit_rank,
            "mrr": 1/first_hit_rank if first_hit_rank else 0,
            "retrieved": retrieved_texts,
        })

    return results


def print_retrieval_report(results):
    """평가 결과 출력"""
    print("\n" + "="*80)
    print("RETRIEVAL 평가 결과")
    print("="*80)

    total_hits = sum(1 for r in results if r["hit"])
    avg_mrr = sum(r["mrr"] for r in results) / len(results)

    print(f"\n📊 전체 지표:")
    print(f"   Hit Rate @ 5: {total_hits}/{len(results)} ({total_hits/len(results)*100:.1f}%)")
    print(f"   평균 MRR: {avg_mrr:.3f}")

    print(f"\n📋 개별 결과:")
    for r in results:
        status = "✅" if r["hit"] else "❌"
        print(f"\n{status} 질문: {r['question']}")
        print(f"   기대 주제: {r['expected_topic']}")
        print(f"   Hit: {r['hit']}, First Hit Rank: {r['first_hit_rank']}")
        print(f"   Top-3 검색 결과:")
        for doc in r["retrieved"][:3]:
            print(f"      [{doc['rank']}] (score: {doc['score']:.3f}) {doc['text'][:80]}...")


# =============================================================================
# 4. LLM-as-Judge 평가 (선택적)
# =============================================================================

def evaluate_with_llm_judge(question, context, answer):
    """LLM으로 답변 품질 평가"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""다음 질문에 대한 답변의 품질을 평가해주세요.

질문: {question}

제공된 컨텍스트:
{context}

생성된 답변:
{answer}

평가 기준:
1. 관련성 (1-5): 답변이 질문과 관련 있는가?
2. 정확성 (1-5): 컨텍스트에 기반한 정확한 정보인가?
3. 완전성 (1-5): 질문에 충분히 답변했는가?

JSON 형식으로 평가해주세요:
{{"relevance": X, "accuracy": X, "completeness": X, "comment": "..."}}
"""

    response = llm.invoke(prompt)
    try:
        return json.loads(response.content)
    except:
        return {"error": response.content}


# =============================================================================
# 5. 메인 실행
# =============================================================================

if __name__ == "__main__":
    print("🔄 Vectorstore 로딩...")
    vectorstore = load_vectorstore()

    # 필터 없이 평가
    print("\n" + "="*80)
    print("📌 필터 없이 평가 (원본)")
    print("="*80)
    results_no_filter = evaluate_retrieval(vectorstore, TEST_SET, k=5, use_filter=False)
    print_retrieval_report(results_no_filter)

    # 필터 적용 평가
    print("\n" + "="*80)
    print("📌 필터 적용 평가 (강의 마무리 등 제외)")
    print("="*80)
    results_filtered = evaluate_retrieval(vectorstore, TEST_SET, k=5, use_filter=True)
    print_retrieval_report(results_filtered)

    # 결과 비교
    hits_before = sum(1 for r in results_no_filter if r["hit"])
    hits_after = sum(1 for r in results_filtered if r["hit"])
    mrr_before = sum(r["mrr"] for r in results_no_filter) / len(results_no_filter)
    mrr_after = sum(r["mrr"] for r in results_filtered) / len(results_filtered)

    print("\n" + "="*80)
    print("📊 필터 적용 전후 비교")
    print("="*80)
    print(f"Hit Rate: {hits_before}/5 → {hits_after}/5")
    print(f"MRR: {mrr_before:.3f} → {mrr_after:.3f}")

    # 결과 저장
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"no_filter": results_no_filter, "filtered": results_filtered}, f, ensure_ascii=False, indent=2)
    print("\n💾 결과가 evaluation_results.json에 저장되었습니다.")
