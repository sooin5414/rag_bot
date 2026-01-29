#!/data/edutem/.cache/pypoetry/virtualenvs/rag-bot-vbdTYmCJ-py3.12/bin/python
"""
계층적 검색 시뮬레이션
"""
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Setup
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(
    persist_directory="./chroma_db_with_role",
    embedding_function=embeddings
)
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)

# 테스트 질문들
test_queries = [
    "현재시제가 뭐야?",
    "현재완료 설명해줘",
    "be동사 알려줘",
    "과거형 만드는 법",
]

print("=" * 60)
print("계층적 검색 시뮬레이션")
print("=" * 60)

for query in test_queries:
    print(f"\n🔍 질문: {query}")
    print("-" * 60)

    # 1) Similarity Search (k=5)
    docs = vectorstore.similarity_search(query, k=5)

    print(f"📊 Vector Search Top 5:")
    for i, doc in enumerate(docs, 1):
        topic = doc.metadata.get('topic', 'N/A')
        grammar = doc.metadata.get('grammar_type', 'N/A')
        role = doc.metadata.get('role', 'N/A')
        print(f"  {i}. [{grammar}] {topic} ({role})")

    # 2) Reranking
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    print(f"\n🎯 Reranked Top 3:")
    for i, (doc, score) in enumerate(ranked[:3], 1):
        topic = doc.metadata.get('topic', 'N/A')
        grammar = doc.metadata.get('grammar_type', 'N/A')
        print(f"  {i}. [{grammar}] {topic} (score: {score:.3f})")

    # 3) 토픽 다양성 분석
    unique_topics = list({doc.metadata.get('topic') for doc in docs})
    print(f"\n📌 발견된 고유 토픽 수: {len(unique_topics)}")
    print(f"   {', '.join(unique_topics[:5])}")

    print()
