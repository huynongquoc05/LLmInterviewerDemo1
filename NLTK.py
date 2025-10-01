import os
from dotenv import load_dotenv
import nltk

# ======================
# 1. Chuẩn bị NLTK
# ======================
nltk.download("punkt")
try:
    nltk.download("punkt_tab")  # cần cho NLTK >=3.8.1
except:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.schema import Document

# Load PDF
loader = PyPDFLoader("Chương 2 Biến, hằng và kiểu dữ liệu.pdf")
pages = loader.load()

# Gộp toàn bộ text thành 1 Document duy nhất
full_text = "\n".join([p.page_content for p in pages])
full_doc = [Document(page_content=full_text)]

# ======================
# 3. Chia nhỏ văn bản bằng NLTK
# ======================
text_splitter= NLTKTextSplitter(
    chunk_size=1600,       # độ dài tối đa mỗi chunk (số ký tự)
    chunk_overlap=400,     # số ký tự overlap giữa 2 chunk
    separator="\n\n"       # ký tự tách đoạn (mặc định theo NLTK sentence tokenizer)
)
splitted_docs = []

for doc in full_doc:
    chunks = text_splitter.split_text(doc.page_content)
    for chunk in chunks:
        splitted_docs.append(
            {
                "page_content": chunk,
                "metadata": doc.metadata,  # giữ metadata (trang số, v.v.)
            }
        )

print(f"✂️ Sau khi chia chunk: {len(splitted_docs)} đoạn")

# ======================
# 4. Khởi tạo Embeddings
# ======================
device = "cuda" if os.environ.get("USE_GPU", "1") == "1" else "cpu"
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large-instruct",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
)

# ======================
# 5. Tạo vector store từ text
# ======================
vectorstore = FAISS.from_texts(
    [d["page_content"] for d in splitted_docs],
    embeddings,
    metadatas=[d["metadata"] for d in splitted_docs],
)

# Save to disk (create folder if not exists)
save_path = "vector_db2chunk_nltk"
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)



# # ======================
# # 6. Truy vấn thử
# # ======================
# query = "Đặt tên trong java"
# retriever = vectorstore.as_retriever()
# results = retriever.get_relevant_documents(query)
#
# print("🔍 Kết quả truy vấn:")
# for i, d in enumerate(results, 1):
#     print(f"\n--- Kết quả {i} (Trang {d.metadata.get('page', 'N/A')}) ---")
#     print(d.page_content)
