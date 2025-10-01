from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load PDF
loader = PyPDFLoader("Chương 2 Biến, hằng và kiểu dữ liệu.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
docs = text_splitter.split_documents(documents)

# Danh sách các model và thư mục lưu vector store (đã loại bỏ model cuối)
models = [
    {"name": "intfloat/multilingual-e5-large-instruct", "folder": "vector_db_e5_large"},
    {"name": "hiieu/halong_embedding", "folder": "vector_db_halong"},
    {"name": "AITeamVN/Vietnamese_Embedding", "folder": "vector_db_aiteam", },
    {"name": "dangvantuan/vietnamese-embedding", "folder": "vector_db_dvt"},  # Loại bỏ model này
]
models_need_trust = {
    "dangvantuan/vietnamese-document-embedding",
"Alibaba-NLP/gte-multilingual-base"
}

# Tạo và lưu vector store cho từng model
for model_info in models:
    model_name = model_info["name"]
    save_path = model_info["folder"]

    print(f"Processing model: {model_name}")
    if model_name in models_need_trust:
        # Khởi tạo embedding
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True} , # Normalize cho cosine similarity
            trust_remote_code=True
        )
    else:
        # Khởi tạo embedding
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Normalize cho cosine similarity
        )

    # Tạo vector store với FAISS
    db = FAISS.from_documents(docs, embeddings)

    # Tạo thư mục lưu nếu chưa có
    os.makedirs(save_path, exist_ok=True)

    # Lưu vector store
    db.save_local(save_path)
    print(f"Vector store for {model_name} saved to {save_path}")

print("All vector stores have been created and saved successfully!")