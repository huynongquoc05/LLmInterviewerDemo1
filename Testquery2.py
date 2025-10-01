from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import torch
import gc

# Danh sách vector store và model tương ứng
vector_stores = [
    {"model_name": "intfloat/multilingual-e5-large-instruct", "folder": "vector_db_e5_large"},
    {"model_name": "hiieu/halong_embedding", "folder": "vector_db_halong"},
    {"model_name": "AITeamVN/Vietnamese_Embedding", "folder": "vector_db_aiteam"}
]


# Câu query
query = "Làm thế nào để nhập dữ liệu từ bàn phím trong Java?"

# Test từng vector store
for store_info in vector_stores:
    model_name = store_info["model_name"]
    folder = store_info["folder"]

    print(f"\n=== Testing vector store for {model_name} (folder: {folder}) ===")

    # Giải phóng bộ nhớ GPU trước khi load
    torch.cuda.empty_cache()
    gc.collect()

    # Load embeddings cho model
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},  # Thay bằng 'cpu' nếu GPU lỗi
        encode_kwargs={'normalize_embeddings': True}
    )

    # Load vector store
    db = FAISS.load_local(folder, embeddings=embeddings, allow_dangerous_deserialization=True)


    # 2. as_retriever
    print("\nResults from as_retriever (Top 4):")
    retriever = db.as_retriever(search_kwargs={'k': 4})  # Lấy top 4
    docs = retriever.get_relevant_documents(query)
    for r in docs:
        print(r.page_content[:])  # In 500 ký tự đầu
        print("-" * 80)

print("All vector stores have been tested successfully!")