from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

# Load PDF
loader = PyPDFLoader("java_tutorial.pdf")
documents = loader.load()

# Split text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=150,
    length_function=len,
)
docs = text_splitter.split_documents(documents)

# Embedding
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
# Note: If you want to use a different model, you can change the model_name parameter.

# Vector store (FAISS)
db = FAISS.from_documents(docs, embeddings)

# Save to disk (create folder if not exists)
save_path = "vector_db_e5_large"
os.makedirs(save_path, exist_ok=True)
db.save_local(save_path)