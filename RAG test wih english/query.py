from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

# Load lại FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
db = FAISS.load_local(
    "vector_db_e5_large",
    embeddings,
    allow_dangerous_deserialization=True
)

query = "Java 8 features added"
results = db.similarity_search(query, k=10)  # lấy nhiều hơn để rerank

# --- Cross-Encoder Re-Ranker ---
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")

# Tạo input dạng (query, result)
pairs = [(query, r.page_content) for r in results]
scores = reranker.predict(pairs)

# Gắn score vào kết quả và sắp xếp lại
scored_results = list(zip(results, scores))
scored_results = sorted(scored_results, key=lambda x: x[1], reverse=True)

print("Query:", query)
# In kết quả sau khi rerank
for i, (doc, score) in enumerate(scored_results, 1):
    print(f"--- ReRanked Result {i} (score={score:.4f}) ---")
    print(doc.page_content[:])
    print()
