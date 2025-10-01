# LLMInterviewer: Tạo câu hỏi phỏng vấn, hỏi thí sinh, chấm điểm dựa trên knowledge DB
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from GetApikey import loadapi

# =======================
# 1. Load API & Embedding
# =======================
API_KEY = loadapi()
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")

# =======================
# 2. Load FAISS databases
# =======================
cv = FAISS.load_local("vector_db_csv", embeddings, allow_dangerous_deserialization=True)
knowledge = FAISS.load_local("vector_db2chunk_nltk", embeddings, allow_dangerous_deserialization=True)

# =======================
# 3. Retriever & LLM
# =======================
retriever = knowledge.as_retriever(search_kwargs={"k": 5})

llm = GoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=1
)

# =======================
# 4. Interview Orchestrator
# =======================

import json
import re

def _clean_and_parse_questions_text(raw_text: str):
    """
    Input: raw_text returned từ LLM (có thể kèm ```json``` hoặc text lộn xộn)
    Output: list[str] các câu hỏi sạch
    """
    if not raw_text:
        return []

    text = raw_text.strip()

    # 1) Nếu có khối code fence ```...``` -> lấy phần bên trong
    code_fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    # 2) Cố gắng trích object JSON hoàn chỉnh giữa dấu { ... }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "questions" in parsed and isinstance(parsed["questions"], list):
                return [_sanitize_question(q) for q in parsed["questions"] if isinstance(q, str) and q.strip()]
        except Exception:
            # fallthrough to other strategies
            pass

    # 3) Nếu không parse JSON, cố gắng lấy tất cả chuỗi trong ngoặc kép (")
    quoted = re.findall(r'"([^"]{5,})"', text, flags=re.S)  # lấy các chuỗi ít nhất 5 ký tự
    if quoted:
        return [_sanitize_question(q) for q in quoted]

    # 4) Fallback: lấy các dòng dài và bỏ những dòng cấu trúc JSON đơn thuần
    lines = [ln.strip() for ln in text.splitlines()]
    candidate_lines = []
    for ln in lines:
        if not ln:
            continue
        # loại các dòng thuần là JSON punctuation
        if re.fullmatch(r'[\{\}\[\],]*', ln):
            continue
        # loại dòng như '"questions": [' hoặc 'questions:' hoặc '```json'
        if re.match(r'^(questions|\"questions\"|```json)', ln, flags=re.I):
            continue
        candidate_lines.append(ln)

    # Nếu candidate_lines có dạng danh sách giống "- ...", "1. ..." -> nhóm thành câu hỏi
    # Ghép các dòng bắt đầu bằng số/dấu gạch liền nhau
    questions = []
    buffer = []
    for ln in candidate_lines:
        # nếu dòng bắt đầu bằng số+dot hoặc gạch đầu dòng -> bắt đầu câu hỏi mới
        if re.match(r'^\s*(?:\d+[\).\s-]+|[-•\*]\s+)', ln):
            if buffer:
                questions.append(" ".join(buffer).strip())
                buffer = []
            # loại bỏ phần số/dấu
            ln = re.sub(r'^\s*(?:\d+[\).\s-]+|[-•\*]\s+)', '', ln)
            buffer.append(ln)
        else:
            # nếu dòng rất ngắn (nhỏ hơn 8 ký tự) và có dấu ngoặc/backtick, loại bỏ
            buffer.append(ln)
    if buffer:
        questions.append(" ".join(buffer).strip())

    # Nếu vẫn rỗng, dùng câu dài > 15 ký tự làm câu hỏi
    if not questions:
        questions = [ln for ln in candidate_lines if len(ln) > 15]

    # sanitize each
    return [_sanitize_question(q) for q in questions if q and len(q) > 5]


def _sanitize_question(q: str) -> str:
    """Làm sạch 1 câu hỏi: bỏ backticks/quote, bỏ số thứ tự đầu, trim."""
    s = q.strip()
    # remove surrounding backticks or quotes
    s = re.sub(r'^[`\"]+|[`\"]+$', '', s).strip()
    # remove leading JSON quotes if any
    s = re.sub(r'^\s*"\s*', '', s)
    # remove leading numbering like 1.  or 1) or -
    s = re.sub(r'^\s*\(?\d+\)?[\).\s:-]+\s*', '', s)
    # remove stray trailing commas/brackets
    s = s.rstrip(",;]")
    return s.strip()


def generate_questions(candidate_name: str, topic: str):
    # --- Step 1: Load candidate profile ---
    profile_docs = cv.similarity_search(candidate_name, k=1)
    profile_context = profile_docs[0].page_content if profile_docs else "Không tìm thấy hồ sơ."
    print("=== Candidate Profile ===")
    print(profile_context)

    # --- Step 2: Classify level ---
    classify_prompt = f"""
        Bạn là một Interviewer AI cho kỳ thi vấn đáp cuối kỳ.
        Dựa vào hồ sơ thí sinh sau đây, hãy phân loại trình độ của thí sinh này
        theo một trong các mức dựa vào điểm 40%: Yếu (<5), Trung bình (5-<6.5), Khá(6.5-<8), Giỏi(8-<9), Xuất sắc (9-10).

        Chỉ trả lời đúng 1 từ duy nhất là mức phân loại, không giải thích.

        Hồ sơ thí sinh:
        {profile_context}

        Trình độ:
        """
    level = llm.invoke(classify_prompt).strip()
    print("\n=== Classified Level ===")
    print(level)

    # --- Step 3: Generate 3 questions ---
    knowledge_context = retriever.invoke(topic)
    knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_context])

    generate_prompt = f"""
Bạn là một Interviewer AI.
Tạo **chính xác 3 câu hỏi** phỏng vấn (tiếng Việt) về chủ đề: "{topic}" phù hợp với trình độ ứng viên.

**Yêu cầu bắt buộc**:
- Trả về **CHỈ** **một object JSON thuần** có dạng:
  {{ "questions": ["Câu 1...", "Câu 2...", "Câu 3..."] }}
- KHÔNG kèm lời chào, giải thích, hay code fence (```).
- Câu 1: cơ bản, Câu 2: trung cấp, Câu 3: nâng cao.
- Mỗi câu là 1 chuỗi văn bản hoàn chỉnh, không có dấu chấm phẩy nối nhiều câu.

Tài liệu tham khảo (trích từ knowledge DB):
{knowledge_text}

Hồ sơ ứng viên:
{profile_context}

Trình độ: {level}
"""
    questions_raw = llm.invoke(generate_prompt)

    questions = _clean_and_parse_questions_text(questions_raw)

    # đảm bảo có ít nhất 1..3 câu hỏi; nếu >3 chỉ lấy 3, nếu <3 vẫn trả về những gì có
    if len(questions) > 3:
        questions = questions[:3]
    print("\n=== Generated Questions ===")
    for idx, q in enumerate(questions, start=1):
        print(f"{idx}. {q}")

    return {
        "profile": profile_context,
        "level": level,
        "planned_questions": questions
    },knowledge_text


def run_interview(candidate_name: str, topic: str):
    data,knowledge_text = generate_questions(candidate_name, topic)
    planned = data.get("planned_questions", [])
    asked = []
    answers = []

    if not planned:
        print("⚠ Không có câu hỏi nào được tạo. Kết thúc.")
        return {
            "profile": data["profile"],
            "level": data["level"],
            "planned_questions": [],
            "answers": []
        }

    print("\n--- Bắt đầu phần vấn đáp (hỏi tuần tự 3 câu dự kiến) ---")
    for i, q in enumerate(planned, start=1):
        # in rõ Bot hỏi
        print(f"\n🤖 Interviewer (Bot) - Câu hỏi {i}:")
        print(q)
        # in prompt rõ Thí sinh trả lời
        ans = input("👩‍🎓 Thí sinh trả lời (nhập xong ENTER): ").strip()
        # để đơn giản: nếu user chỉ nhấn Enter, lưu chuỗi rỗng và tiếp tục
        asked.append(q)
        answers.append(ans)

    # Lưu log (chưa chấm)
    interview_log = {
        "profile": data["profile"],
        "level": data["level"],
        "planned_questions": asked,
        "answers": answers
    }

    print("\n=== Interview Summary ===")
    print(json.dumps(interview_log, indent=2, ensure_ascii=False))

    return interview_log

# def eval_during_interview(...,llm, knowledge_text):
#     """
#     Vừa hỏi vừa chấm điểm từng câu.
#     Đánh giá câu trả lời của thí sinh dựa vào knowledge DB. nếu điểm <5 (cảm thấy câu trả lời của thí sinh không tốt),
#     dựa trên tài liệu ngữ cảnh tìm 1 câu khác dễ hơn thay thế.
#     """
#
#
#
#
#     # --- 2. Prompt đánh giá ---
#     eval_prompt = f"""
#     Bạn là giám khảo chấm thi vấn đáp lập trình Java.
#     Câu hỏi: {q}
#     Câu trả lời của thí sinh: {a}
#
#     Tài liệu tham chiếu (knowledge base):
#     {knowledge_text}
#
#     Nhiệm vụ:
#     - So sánh câu trả lời với tài liệu tham chiếu.
#     - Đánh giá mức độ chính xác, đầy đủ, rõ ràng.
#     - Chấm điểm trên thang 0–10.
#     - Nếu điểm <5, dựa trên tài liệu tham chiếu, tìm một câu hỏi khác dễ hơn để thay thế câu hỏi gốc.
#     **Yêu cầu bắt buộc**:
#     - Trả về **CHỈ** **một object JSON thuần** có dạng:
#       {{ "questions": "câu hỏi thay thế" }}
#     - KHÔNG kèm lời chào, giải thích, hay code fence (```).
#     - Câu hỏi thay thế phải dễ hơn câu gốc.
#     - câu là 1 chuỗi văn bản hoàn chỉnh, không có dấu chấm phẩy nối nhiều câu.
#     """
#
#     result = llm.invoke(eval_prompt)
#     alternative_questions = _clean_and_parse_questions_text(result)


def evaluate_answers(interview_result: dict, retriever, llm,topic: str):
    """
    Đánh giá câu trả lời của thí sinh dựa vào knowledge DB.
    Return: list kết quả [{question, answer, score, analysis}]
    """

    evaluations = []
    knowledge_context = retriever.invoke(topic)
    knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_context])
    for i, (q, a) in enumerate(zip(interview_result["planned_questions"], interview_result["answers"])):


        # --- 2. Prompt đánh giá ---
        eval_prompt = f"""
        Bạn là giám khảo chấm thi vấn đáp lập trình Java.
        Câu hỏi: {q}
        Câu trả lời của thí sinh: {a}

        Tài liệu tham chiếu (knowledge base):
        {knowledge_text}

        Nhiệm vụ:
        - So sánh câu trả lời với tài liệu tham chiếu.
        - Đánh giá mức độ chính xác, đầy đủ, rõ ràng.
        - Chấm điểm trên thang 0–10.
        - Trả về JSON với cấu trúc:
        {{
          "score": <điểm từ 0 đến 10>,
          "analysis": "<nhận xét ngắn gọn>"
        }}
        """

        result = llm.invoke(eval_prompt)

        # --- 3. Lưu kết quả ---
        evaluations.append({
            "question": q,
            "answer": a,
            "evaluation": result
        })

    return evaluations

# === Ví dụ chạy ===
# run_interview("Trần Thị Bình", "Đặt tên trong java")

# =======================
# 5. Run demo
# =======================
if __name__ == "__main__":
    candidate_name = "Lê Thị Trang,QTKD2"
    topic = "Đặt tên trong Java"
    # run_interview(candidate_name, topic)
    interview_result = run_interview(candidate_name, topic)
    evaluation_result = evaluate_answers(interview_result, retriever, llm, topic)
    for ev in evaluation_result:
        print("\n=== Evaluation ===")
        print(f"Q: {ev['question']}")
        print(f"A: {ev['answer']}")
        print(f"Result: {ev['evaluation']}")