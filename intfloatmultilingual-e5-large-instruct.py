# Load from disk
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

db=FAISS.load_local("vector_db2chunk_nltk", embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct", )
,allow_dangerous_deserialization=True)
# # # Example query
# query= """Nhập dữ liệu từ bàn phím trong Java
# # """
# #
# #
# results = db.similarity_search_with_score(query, k=10)
# print("query:", query)
# for r, score in results:
#     print(len(r.page_content))
#     print(f"Score: {score:.4f}")
#     print(r.page_content[:])
#     print("-" * 80)

from difflib import SequenceMatcher

def is_similar(a, b, threshold=0.9):
    """So sánh hai đoạn text, trả về True nếu giống nhau trên ngưỡng."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def deduplicate_texts(texts, threshold=0.7):
    """Loại bỏ các đoạn trùng hoặc gần giống nhau."""
    unique_texts = []
    for t in texts:
        t_clean = t.strip()
        if not any(is_similar(t_clean, u, threshold) for u in unique_texts):
            unique_texts.append(t_clean)
    return unique_texts

exeptipnPrompts=['NGOẠI LỆ',
                'Xử lý ngoại lệ',
                'Dây chuyền xử lý ngoại lệ',
                'Phân cấp ngoại lệ',
                'Ném ngoại lệ và tạo lớp ngoại lệ'
                'Xử lý ngoại lệ bắt buộc'
                ]

datatypePrompts=['Kiểu dữ liệu cơ sở','Kiểu dữ liệu "Gói" (Wrapper)', 'nhập xuất các kiểu dữ liệu trong java']

# StringPrompt=[
#     'Chuỗi ký tự',
#     'Khởi tạo chuỗi ký tự',
# 'phương thức xử lý chuỗi kỹ tự'
# ]
#
# all_results = []
#
# for query in exeptipnPrompts:
#     results = db.similarity_search_with_score(query, k=5)
#     for r, score in results:
#         all_results.append(r.page_content)
# # retreiver=db.as_retriever(search_type="similarity", search_kwargs={"k":5})
# # all_results = retreiver.invoke_batch(exeptipnPrompts)
# # # === Lọc trùng lặp ===
# # filtered_results = deduplicate_texts(all_results, threshold=0.9)
# #
# # combined_context = "\n".join(filtered_results)
#
# combined_context = "\n".join(all_results)
#
#
# from GetApikey import loadapi
#
# API_KEY=loadapi()
#
# # LLM Google Gemini (text-only)
# llm = GoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     google_api_key=API_KEY,
#     temperature=0.5
# )
#
# from langchain.prompts import PromptTemplate
#
# # Prompt để tổng hợp nội dung
# prompt_template = """
# Bạn là một trợ lý AI. Dưới đây là các đoạn nội dung được trích xuất từ nhiều truy vấn liên quan đến **ngoại lệ trong lập trình**.
#
# Nội dung:
# {context}
#
# Hãy viết một bản tổng hợp ngắn gọn, có cấu trúc rõ ràng,
# trình bày đầy đủ khái niệm, cách xử lý, phân cấp,
# dây chuyền xử lý, ném ngoại lệ, tạo lớp ngoại lệ,
# và các lưu ý quan trọng về ngoại lệ trong lập trình.
# không thêm thông tin không có trong nội dung đã cho.
# """
#
# prompt = PromptTemplate(
#     input_variables=["context"],
#     template=prompt_template
# )
#
#
# final_input = prompt.format(context=combined_context)
#
# # Gọi LLM để sinh nội dung tổng hợp
# summary = llm.invoke(final_input)
#
# print("📌 Bản tổng hợp về ngoại lệ:\n")
# print(summary)


# StringPrompt=[
#     'Chuỗi ký tự',
#     'Khởi tạo chuỗi ký tự',
# 'phương thức xử lý chuỗi kỹ tự'
# ]
#
all_results = []

for query in datatypePrompts:
    results = db.similarity_search_with_score(query, k=5)
    for r, score in results:
        print(r)
        all_results.append(r.page_content)
# === Lọc trùng lặp ===
filtered_results = deduplicate_texts(all_results, threshold=0.9)

combined_context = "\n".join(filtered_results)


from GetApikey import loadapi

API_KEY=loadapi()

# LLM Google Gemini (text-only)
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.5
)

from langchain.prompts import PromptTemplate

# Prompt để tổng hợp nội dung
prompt_template = """
Bạn là một trợ lý AI. Dưới đây là các đoạn nội dung được trích xuất từ nhiều truy vấn liên quan đến **kiểu dữ liệu trong java**.

Nội dung:
{context}
Nếu thấy thông tin không liên quan, hãy bỏ qua.
Hãy viết một bản tổng hợp ngắn gọn, có cấu trúc rõ ràng,
Không bỏ sót thông tin quan trọng nào.
Cũng không thêm thông tin không có trong nội dung đã cho.
"""

prompt = PromptTemplate(
    input_variables=["context"],
    template=prompt_template
)


final_input = prompt.format(context=combined_context)

# Gọi LLM để sinh nội dung tổng hợp
summary = llm.invoke(final_input)

print("📌 Bản tổng hợp về chuỗi ký tự:\n")
print(summary)
