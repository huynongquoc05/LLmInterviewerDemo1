from pymongo import MongoClient


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


import requests
import os
from GetApikey import get_api_key_elevenlab


# extension.py
from GetApikey import get_api_key_elevenlab
import requests

def generate_voice_ElevenLab(text, output_path="output.mp3"):
    """
    Sinh voice từ text bằng ElevenLabs.
    Trả về đường dẫn file nếu thành công, None nếu lỗi (vd hết limit).
    """
    API_KEY = get_api_key_elevenlab()
    if not API_KEY:
        return None

    url = "https://api.elevenlabs.io/v1/text-to-speech/pqHfZKP75CvOlQylNhV4"
    headers = {"xi-api-key": API_KEY, "Content-Type": "application/json"}
    data = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {
            "stability": 0.8,
            "similarity_boost": 0.8
        },
        "voice_speed": 1.1
    }

    try:
        res = requests.post(url, json=data, headers=headers)
        if res.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(res.content)
            return output_path
        elif res.status_code == 429:
            print("⚠️ Hết limit ElevenLabs.")
            return None
        else:
            print(f"⚠️ Lỗi ElevenLabs: {res.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Lỗi gửi request ElevenLabs: {e}")
        return None


def get_vectorstore_chunks(vectorstore_id, mongo_uri="mongodb://localhost:27017/"):
    """
    Trích xuất nội dung từng chunk trong vectorstore
    - Đọc metadata vectorstore từ MongoDB
    - Tự động chọn đúng model embedding
    - Trả về danh sách các đoạn văn bản (chunks)
    """
    from bson import ObjectId

    # Kết nối DB
    client = MongoClient(mongo_uri)
    db = client["interviewer_ai"]
    vs = db["vectorstores"].find_one({"_id": ObjectId(vectorstore_id)})

    if not vs:
        raise ValueError(f"Vectorstore {vectorstore_id} not found")

    # Lấy thông tin từ metadata
    path = vs.get("vectorstore_path")
    if not path:
        raise ValueError(f"Vectorstore {vectorstore_id} missing vectorstore_path")

    # Lấy model embedding chính xác từ metadata
    model_name = vs.get("model_name") or vs.get("model_info", {}).get("name")
    if not model_name:
        model_name = "intfloat/multilingual-e5-large-instruct"  # fallback an toàn

    # Khởi tạo embedding và load FAISS
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vs_local = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    # Trích xuất nội dung các chunk
    docs = []
    for i, doc in enumerate(vs_local.docstore._dict.values()):
        docs.append({
            "index": i + 1,
            "content": doc.page_content,
            "metadata": doc.metadata or {},
        })

    return docs
