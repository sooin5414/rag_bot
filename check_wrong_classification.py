"""
조동사로 잘못 분류된 항목 확인
"""
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

collection = vectorstore._collection
all_results = collection.get(include=["metadatas", "documents"])

print("🔍 조동사로 분류된 항목 중 의심스러운 것 찾기\n")

for i, (meta, doc) in enumerate(zip(all_results['metadatas'], all_results['documents'])):
    if meta.get('grammar_type') == '조동사':
        topic = meta.get('topic', '')
        content = doc[:150]

        # be동사 관련 키워드 체크
        suspicious_keywords = ['is', 'am', 'are', 'was', 'were', 'be동사', 'be 동사']
        if any(keyword in topic.lower() or keyword in content.lower() for keyword in suspicious_keywords):
            print(f"❌ 의심: ID {i}")
            print(f"   Topic: {topic}")
            print(f"   Role: {meta.get('role')}")
            print(f"   Content: {content}...")
            print()

print("\n📊 조동사 분류 전체 토픽 리스트:")
modal_topics = {}
for meta in all_results['metadatas']:
    if meta.get('grammar_type') == '조동사':
        topic = meta.get('topic', 'unknown')
        modal_topics[topic] = modal_topics.get(topic, 0) + 1

for topic, count in sorted(modal_topics.items(), key=lambda x: x[1], reverse=True):
    marker = "⚠️" if any(kw in topic.lower() for kw in ['is', 'am', 'are', 'be']) else "✅"
    print(f"{marker} {topic}: {count}개")
