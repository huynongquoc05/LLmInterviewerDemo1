import os

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Đọc dữ liệu CSV
df = pd.read_csv("danhsach_thisinh.csv")

# 2. Chuyển thành text mô tả từng thí sinh
def row_to_text(row):
    return f"Họ tên: {row['Tên']}, Lớp: {row['Lớp']}, " \
           f"Chuyên ngành: {row['Chuyên ngành']}, " \
           f"Điểm chuyên cần: {row['Điểm chuyên cần']}, " \
           f"Điểm 40%: {row['Điểm 40%']}, " \


texts = [row_to_text(r) for _, r in df.iterrows()]

# 3. Khởi tạo embedding model
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# 4. Tạo vector store từ text
vectorstore = FAISS.from_texts(texts, embeddings)
save_path = "vector_db_csv"
os.makedirs(save_path, exist_ok=True)
vectorstore.save_local(save_path)
print(f"Vector store saved to {save_path}")
