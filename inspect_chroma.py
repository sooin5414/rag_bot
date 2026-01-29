"""
ChromaDB 내용 확인 스크립트
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ChromaDB 로드
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# 전체 문서 개수
collection = vectorstore._collection
total_count = collection.count()
print(f"📊 총 문서 개수: {total_count}")

# 샘플 데이터 조회 (처음 5개)
results = collection.get(
    limit=5,
    include=["documents", "metadatas"]
)

print(f"\n📄 샘플 데이터 (처음 5개):\n")
for i, (doc, meta) in enumerate(zip(results['documents'], results['metadatas']), 1):
    print(f"--- Document {i} ---")
    print(f"Topic: {meta.get('topic', 'N/A')}")
    print(f"Role: {meta.get('role', 'N/A')}")
    print(f"Grammar Type: {meta.get('grammar_type', 'N/A')}")
    print(f"Summary: {meta.get('summary', 'N/A')}")
    print(f"Video URL: {meta.get('video_url', 'N/A')}")
    print(f"Time: {meta.get('start_time', 'N/A')}s ~ {meta.get('end_time', 'N/A')}s")
    print(f"Keywords: {meta.get('keywords', 'N/A')}")
    print(f"Content: {doc[:100]}...")
    print()

# Topic별 통계
print("\n📌 Topic별 문서 개수:")
all_results = collection.get(include=["metadatas"])
topics = {}
roles = {}
grammar_types = {}

for meta in all_results['metadatas']:
    topic = meta.get('topic', 'unknown')
    role = meta.get('role', 'unknown')
    grammar = meta.get('grammar_type', 'unknown')

    topics[topic] = topics.get(topic, 0) + 1
    roles[role] = roles.get(role, 0) + 1
    grammar_types[grammar] = grammar_types.get(grammar, 0) + 1

# 상위 10개 토픽
sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
for topic, count in sorted_topics[:10]:
    print(f"  {topic}: {count}개")

print(f"\n🎭 Role별 분포:")
for role, count in sorted(roles.items(), key=lambda x: x[1], reverse=True):
    print(f"  {role}: {count}개")

print(f"\n📚 Grammar Type별 분포:")
for grammar, count in sorted(grammar_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {grammar}: {count}개")
