


from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.schema import Document

# Load PDF
loader = PyPDFLoader("Chương 2 Biến, hằng và kiểu dữ liệu.pdf")
pages = loader.load()

# Gộp toàn bộ text thành 1 Document duy nhất
full_text = "\n".join([p.page_content for p in pages])
full_doc = [Document(page_content=full_text)]

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
docs = text_splitter.split_documents(full_doc)

# # Embedding
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# # Note: If you want to use a different model, you can change the model_name parameter.
#
# # Vector store (FAISS)
# db = FAISS.from_documents(docs, embeddings)
#
# # Save to disk (create folder if not exists)
# save_path = "vector_db"
# os.makedirs(save_path, exist_ok=True)
# db.save_local(save_path)


# # model khác
# model_name = "keepitreal/vietnamese-sbert"
# # Embedding
# embeddings = HuggingFaceEmbeddings(model_name=model_name)
# # Note: If you want to use a different model, you can change the model_name parameter.
#
# # Vector store (FAISS)
# db = FAISS.from_documents(docs, embeddings)
#
# # Save to disk (create folder if not exists)
# save_path = "vector_db1"
# os.makedirs(save_path, exist_ok=True)
# db.save_local(save_path)

# model khác
model_name = "intfloat/multilingual-e5-large-instruct"
# Embedding
embeddings = HuggingFaceEmbeddings(model_name=model_name)
# Note: If you want to use a different model, you can change the model_name parameter.

# Vector store (FAISS)
db = FAISS.from_documents(docs, embeddings)

# Save to disk (create folder if not exists)
save_path = "vector_db2b"
os.makedirs(save_path, exist_ok=True)
db.save_local(save_path)