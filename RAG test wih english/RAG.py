from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
import keyboard

from GetApikey import loadapi

API_KEY = loadapi()

# Load lại FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("vector_db_el", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 5})

# LLM Google Gemini
llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.3
)

# Prompt cho RAG chat
prompt_template = """
Bạn là một trợ lý AI.
Ai hỏi bạn là ai, hãy trả lời bạn là Java RAG Bot, chuyên hỗ trợ các câu hỏi về Java.
Khi có câu hỏi hãy thử tìm câu trả lời trong phần context bên dưới bằng những từ khóa khác nhau nhằm tìm ra k quả chính xác, đầy đủ nhất.
Hãy sử dụng **chỉ thông tin trong context** và lịch sử hội thoại để trả lời câu hỏi bằng tiếng Việt. 
Câu trả lời cần:
- Ngắn gọn, rõ ràng, có đủ ý.
- Không thêm thông tin ngoài context.
- Nếu context không liên quan hoặc thiếu, hãy nói: 
  "Thông tin tìm được không liên quan hoặc chưa đủ để trả lời đầy đủ."

Context: {context}

Câu hỏi: {question}

Trả lời:
"""
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Memory lưu lịch sử hội thoại
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt}
)

# Vòng lặp chat
print("💬 Chat với Java RAG Bot (gõ 'exit' để thoát)\n")
while True:
    query = input("❓Bạn: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Kết thúc chat.")
        break
    # exit khi nhấn 'Esc'
    if keyboard.is_pressed('esc'):
        print("👋 Kết thúc chat.")
        break

    result = qa_chain.invoke({"question": query})
    print("🤖 Bot:", result["answer"])


