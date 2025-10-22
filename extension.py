def summarize_knowledge_with_llm(knowledge_text: str, topic: str, outline: list[str], llm):
    """
    Dùng LLM để tóm tắt và đánh giá chất lượng nguồn tài liệu RAG.
    """
    if not knowledge_text or len(knowledge_text.strip()) == 0:
        return {"summary": "(Không có tài liệu)", "quality_report": "Không có nội dung để đánh giá."}

    outline_str = "\n".join(f"- {item}" for item in outline or [])

    prompt = f"""
    Bạn là chuyên gia đào tạo Java, hãy giúp đánh giá và tóm tắt tài liệu phỏng vấn sau.

    CHỦ ĐỀ: {topic}
    OUTLINE (mục tiêu kiến thức): 
    {outline_str}

    --- TÀI LIỆU TRUY VẤN ---
    {knowledge_text}  # Giới hạn để tránh token overflow

    HÃY TRẢ VỀ Bản tóm tắt 
    """

    result = llm.invoke(prompt)
    return result


import os
import pandas as pd
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


def build_cv_vectorstore_from_candidates(candidates, base_dir="vectorstores/cv"):
    """
    Tạo vectorstore FAISS cho danh sách thí sinh.

    Args:
        candidates (list[dict] | str): danh sách dict hoặc path CSV.
        base_dir (str): thư mục gốc lưu FAISS.
    Returns:
        str: Đường dẫn vectorstore đã lưu.
    """
    os.makedirs(base_dir, exist_ok=True)

    # 1️⃣ Load data
    if isinstance(candidates, str):
        df = pd.read_csv(candidates)
    else:
        df = pd.DataFrame(candidates)

    # 2️⃣ Chuẩn hóa nội dung mô tả thí sinh
    def row_to_text(row):
        # Duyệt toàn bộ cột => mô tả linh hoạt
        parts = [f"{col}: {val}" for col, val in row.items()]
        return ", ".join(parts)

    texts = [row_to_text(r) for _, r in df.iterrows()]

    # 3️⃣ Sinh embedding
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

    # 4️⃣ Tạo FAISS vectorstore
    vectorstore = FAISS.from_texts(texts, embeddings)

    # 5️⃣ Lưu lại với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(base_dir, f"cv_{timestamp}")
    vectorstore.save_local(save_path)

    print(f"✅ CV Vectorstore saved to {save_path}")
    return save_path
