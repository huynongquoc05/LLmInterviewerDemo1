# Load from disk
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

db=FAISS.load_local("vector_db2", embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", )
,allow_dangerous_deserialization=True)
# Example query
query= "Nhập dữ liệu từ bàn phím trong java"

# results = db.similarity_search_with_score(query, k=3)
# print("query:", query)
# for r, score in results:
#     print(f"Score: {score:.4f}")
#     print(r.page_content[:500])
#     print("-" * 80)

# Example query 2
retriewer=db.as_retriever()
docs=retriewer.get_relevant_documents(query)
for r in docs:
    print(r.page_content)
    print("-" * 80)




