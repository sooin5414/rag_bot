"""
ChromaDB에 저장된 Grammar Type 상세 확인
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

# Grammar Type별로 샘플 출력
grammar_samples = {}
for meta, doc in zip(all_results['metadatas'], all_results['documents']):
    grammar = meta.get('grammar_type', 'unknown')
    topic = meta.get('topic', 'unknown')

    if grammar not in grammar_samples:
        grammar_samples[grammar] = []

    if len(grammar_samples[grammar]) < 3:  # 각 타입당 3개 샘플만
        grammar_samples[grammar].append({
            'topic': topic,
            'role': meta.get('role', 'unknown'),
            'summary': meta.get('summary', '')[:50],
            'content': doc[:80]
        })

print("📚 Grammar Type별 샘플\n")
for grammar_type, samples in sorted(grammar_samples.items()):
    count = sum(1 for m in all_results['metadatas'] if m.get('grammar_type') == grammar_type)
    print(f"🔹 {grammar_type} ({count}개)")
    for i, sample in enumerate(samples, 1):
        print(f"  {i}. Topic: {sample['topic']}")
        print(f"     Role: {sample['role']}")
        print(f"     Summary: {sample['summary']}")
        print(f"     Content: {sample['content']}...")
        print()
