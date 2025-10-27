# AdaptiveInterviewer v2.0: Refactored với Clean Architecture
import datetime
import hashlib
import re
import json
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Callable
from enum import Enum
from pymongo import MongoClient
from bson import ObjectId
import html


from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from GetApikey import loadapi


# =======================
# 1. Enums & Data Classes
# =======================

class Level(Enum):
    YEU = "yeu"
    TRUNG_BINH = "trung_binh"
    KHA = "kha"
    GIOI = "gioi"
    XUAT_SAC = "xuat_sac"


class QuestionDifficulty(Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"

class InterviewPhase(Enum):
    WARMUP = "warmup"          # Giới thiệu, làm quen
    TECHNICAL = "technical"     # Phỏng vấn chuyên môn
    CLOSING = "closing"         # Kết thúc

@dataclass
class QuestionAttempt:
    question: str
    answer: str
    score: float
    analysis: str
    difficulty: QuestionDifficulty
    timestamp: str
    question_hash: Optional[str] = None



@dataclass
class InterviewConfig:
    """Configurable interview parameters"""
    threshold_high: float = 7.0
    threshold_low: float = 4.0
    max_attempts_per_level: int = 2
    max_total_questions: int = 8
    max_upper_level: int = 2
    llm_temperature: float = 0.5
    max_memory_turns: int = 6

    difficulty_map: Dict[Level, List[QuestionDifficulty]] = field(default_factory=lambda: {
        Level.YEU: [QuestionDifficulty.VERY_EASY, QuestionDifficulty.EASY],
        Level.TRUNG_BINH: [QuestionDifficulty.EASY, QuestionDifficulty.EASY],
        Level.KHA: [QuestionDifficulty.MEDIUM, QuestionDifficulty.HARD],
        Level.GIOI: [QuestionDifficulty.MEDIUM, QuestionDifficulty.VERY_HARD],
        Level.XUAT_SAC: [QuestionDifficulty.MEDIUM, QuestionDifficulty.VERY_HARD],
    })


@dataclass
class InterviewState:
    candidate_name: str
    profile: str
    level: Level
    topic: str
    current_difficulty: QuestionDifficulty
    attempts_at_current_level: int
    max_attempts_per_level: int
    total_questions_asked: int
    max_total_questions: int
    upper_level_reached: int
    history: List[QuestionAttempt]
    is_finished: bool
    outline: Optional[List[str]] = None
    knowledge_text: Optional[str] = None
    final_score: Optional[float] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())

    # ✅ NEW FIELDS
    current_phase: InterviewPhase = InterviewPhase.WARMUP  # Bắt đầu từ warmup
    warmup_questions_asked: int = 0
    max_warmup_questions: int = 2  # 1-2 câu làm quen
    candidate_context: Optional[str] = None  # Tóm tắt về thí sinh từ profile
    outline_summary: Optional[str] = None  # Tóm tắt đánh giá outline tài liệu


# =======================
# 2. Utility Functions
# =======================

def classify_level_from_score(score_40: float) -> Level:
    """Phân loại level dựa trên điểm 40%"""
    if score_40 < 5.0:
        return Level.YEU
    elif score_40 <= 6.5:
        return Level.TRUNG_BINH
    elif score_40 <= 8.0:
        return Level.KHA
    elif score_40 <= 9.0:
        return Level.GIOI
    else:
        return Level.XUAT_SAC


def get_initial_difficulty(level: Level, config: InterviewConfig) -> QuestionDifficulty:
    """Lấy độ khó ban đầu cho level"""
    return config.difficulty_map[level][0]


def calculate_question_hash(question: str) -> str:
    """Calculate hash của câu hỏi để detect duplicate"""
    return hashlib.md5(question.encode()).hexdigest()

import re
import json
import html


def _sanitize_question(q: str) -> str:
    """Làm sạch chuỗi câu hỏi khỏi ký tự thừa, dấu số thứ tự, backtick..."""
    s = str(q or "").strip()
    s = re.sub(r'^[`\"]+|[`\"]+$', '', s)
    s = re.sub(r'^\s*"\s*', '', s)
    s = re.sub(r'^\s*\(?\d+\)?[\).\s:-]+\s*', '', s)
    s = s.rstrip(",;}]")
    return s.strip()


def _extract_fallback_question(text: str) -> str:
    """Cố gắng trích câu hỏi nếu JSON lỗi."""
    # Thử bắt đoạn "question": "..."
    m = re.search(r'"question"\s*:\s*"([\s\S]+?)"\s*}', text)
    if m:
        return m.group(1)
    # Nếu không có, lấy dòng dài nhất
    quoted = re.findall(r'"([^"]{20,})"', text, flags=re.S)
    if quoted:
        return quoted[0]
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 30]
    return max(lines, key=len) if lines else text


# def _convert_newlines_to_br(text: str) -> str:
#     """Thay \n thành <br> cho định dạng hiển thị rõ ràng."""
#     text = text.replace("\\n", "\n")
#     text = re.sub(r'\n{2,}', '<br><br>', text)
#     text = re.sub(r'\n', '<br>', text)
#     return text


def _clean_and_parse_json_response(raw_text: str, expected_keys: list[str] = None) -> dict:
    """
    Xử lý phản hồi từ LLM:
    - Ưu tiên parse JSON chuẩn.
    - Nếu lỗi → fallback sang trích chuỗi thủ công.
    - Giữ nguyên HTML trong <pre><code>.
    """
    if not raw_text:
        return {}

    text = raw_text.strip()

    # 1️⃣ Gỡ code fence nếu có
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = text.rstrip("`").strip("`").strip()

    # 2️⃣ Lấy phần JSON chính
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        q = _extract_fallback_question(text)
        return {"question": _sanitize_question(q)}

    json_str = text[start:end + 1]

    # 3️⃣ Parse JSON an toàn
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        # Fallback: trích chuỗi "question": "..."
        q = _extract_fallback_question(json_str)
        return {"question": _sanitize_question(q)}
    except Exception as e:
        print("⚠️ Lỗi parse JSON:", e)
        q = _extract_fallback_question(json_str)
        return {"question": _sanitize_question(q)}

    # 4️⃣ Chuẩn hóa câu hỏi
    if isinstance(parsed, dict) and "question" in parsed:
        q = parsed["question"]
        q = _sanitize_question(q)

        return {"question": q}

    # 5️⃣ Fallback cuối
    q = _extract_fallback_question(text)
    return {"question": _sanitize_question(q)}




# def _clean_and_parse_single_question(raw_text: str, code_snippet: str = None) -> str:
#     """Fallback: parse text thô nếu không có JSON."""
#     if not raw_text:
#         return ""
#
#     text = raw_text.strip()
#     q = _extract_fallback_question(text)
#     q = html.escape(_sanitize_question(q))
#     q = _convert_newlines_to_br(q)
#
#     if code_snippet:
#         html_snippet = f"<pre><code class='language-java'>{html.escape(code_snippet)}</code></pre>"
#         q += "<br><br>" + html_snippet
#     return q


def _parse_evaluation_response(raw_text: str) -> dict:
    """
    Parse JSON kết quả chấm điểm từ LLM, ví dụ:
    ```json
    {"score": 9.5, "analysis": "Giải thích hợp lý."}
    ```
    """
    if not raw_text:
        return {}

    text = raw_text.strip()

    # Loại bỏ code block (```json ... ```)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = text.rstrip("`").strip("`").strip()

    # Tìm đoạn JSON trong chuỗi
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1:
        json_str = text[start:end+1]
        try:
            return json.loads(json_str)
        except Exception as e:
            print("⚠️ Parse JSON lỗi:", e)
            return {}

    return {}

# =======================
# 3. Memory Manager
# =======================

class ConversationMemory:
    """Quản lý conversation history"""

    def __init__(self, max_turns: int = 6):
        self.memory: List[Dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str):
        """Thêm một đoạn hội thoại"""
        self.memory.append({"role": role, "content": content})
        self.memory = self.memory[-self.max_turns:]

    def build_prompt(self) -> str:
        """Ghép memory thành đoạn hội thoại"""
        if not self.memory:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.memory])

    def clear(self):
        """Xóa memory"""
        self.memory = []


# =======================
# PATCH 3: Thêm WarmupManager Component
# =======================
# Thêm sau class ConversationMemory

class WarmupManager:
    """Component quản lý giai đoạn warm-up"""

    def __init__(self, llm: GoogleGenerativeAI):
        self.llm = llm

    def generate_warmup_question(
            self,
            candidate_name: str,
            candidate_context: str,
            topic: str,
            warmup_count: int
    ) -> str:
        """
        Tạo câu hỏi warm-up dựa trên context của thí sinh

        Args:
            candidate_name: Tên thí sinh
            candidate_context: Tóm tắt về thí sinh (từ CV/profile)
            topic: Chủ đề phỏng vấn
            warmup_count: Câu hỏi warm-up thứ mấy (0, 1, 2...)
        """

        warmup_templates = {
            0: f"""
Bạn là interviewer AI thân thiện và chuyên nghiệp.

THÔNG TIN THÍ SINH:
{candidate_context}

Hãy chào hỏi và giới thiệu về buổi phỏng vấn. Bao gồm:
1. Chào thí sinh bằng tên
2. Giới thiệu chủ đề: "{topic}"
3. Đặt 1 câu hỏi warm-up nhẹ nhàng về kinh nghiệm/sở thích liên quan đến {topic}

YÊU CẦU:
- Thân thiện, tạo không khí thoải mái
- Câu hỏi dễ trả lời, KHÔNG cần kiến thức sâu
- Giúp thí sinh "làm nóng máy" trước khi vào phần chuyên môn

OUTPUT: JSON
{{"question": "lời chào + câu hỏi warm-up"}}
""",
            1: f"""
            Bạn là interviewer AI thân thiện và chuyên nghiệp, đang tiếp tục phần warm-up
            với thí sinh {candidate_name}.

            THÔNG TIN THÍ SINH:
            {candidate_context}

            Câu hỏi trước đã giúp bạn hiểu sơ qua về ứng viên.
            Bây giờ, hãy đặt thêm 1 câu hỏi warm-up mới về:
            - Động lực học {topic}
            - Mục tiêu nghề nghiệp
            - Cách ứng viên có thể áp dụng {topic} trong học tập hoặc công việc

            YÊU CẦU:
            - KHÔNG chào lại thí sinh (không dùng "Chào..." ở đầu)
            - Có thể bắt đầu bằng câu chuyển tiếp tự nhiên, ví dụ:
              - "Cảm ơn chia sẻ rất thú vị của bạn, ..."
              - "Nghe thật hay, tiếp theo tôi muốn hỏi thêm..."
              - "Rất thú vị, vậy thì..."
            - Giữ giọng thân thiện, ngắn gọn, không đi sâu kỹ thuật
            - Không cần nói lại chủ đề hoặc giới thiệu phỏng vấn nữa

            OUTPUT: JSON
            {{"question": "câu hỏi warm-up thứ 2, có lời chuyển mượt"}}
            """

        }

        prompt = warmup_templates.get(warmup_count, warmup_templates[1])

        result = self.llm.invoke(prompt)
        parsed = _clean_and_parse_json_response(result)

        return parsed.get("question", f"Xin chào {candidate_name}! Bạn đã sẵn sàng cho buổi phỏng vấn chưa?")

    def extract_candidate_context(self, profile: str) -> str:
        """
        Trích xuất thông tin quan trọng từ CV/profile để LLM hiểu về thí sinh

        Returns:
            Tóm tắt ngắn gọn về thí sinh (200-300 từ)
        """
        # Bạn có thể dùng LLM để summarize hoặc dùng regex
        # Ví dụ đơn giản:
        lines = profile.split('\n')
        summary_lines = []

        keywords = ['tên', 'lớp', 'điểm', 'kỹ năng', 'dự án', 'kinh nghiệm', 'sở thích']

        for line in lines[:15]:  # Lấy 15 dòng đầu
            if any(kw in line.lower() for kw in keywords):
                summary_lines.append(line)

        context = '\n'.join(summary_lines[:10])  # Max 10 dòng
        return context if context else profile[:500]  # Fallback: 500 ký tự đầu


# =======================
# 4. Question Generator
# =======================

class QuestionGenerator:
    """Component chuyên generate câu hỏi"""
    def __init__(self, llm: GoogleGenerativeAI, config: InterviewConfig):
        self.llm = llm
        self.config = config
        self.asked_questions: set = set()



    def validate_question(self, question: str, topic: str) -> bool:
        """Validate câu hỏi có hợp lệ không"""
        if len(question) < 20 or len(question) > 1000:
            return False
        if "Không đủ dữ liệu" in question:
            return False
        return True

    def generate_with_context(
            self,
            topic: str,
            difficulty: QuestionDifficulty,
            knowledge_text: str,
            memory: ConversationMemory,
            candidate_context: str,  # ✅ NEW
            context: str = "",outline_summary: str = ""

    ) -> str:
        """
        Generate câu hỏi có nhận thức về thí sinh
        """

        difficulty_descriptions = {
            QuestionDifficulty.VERY_EASY: (
                "rất cơ bản – kiểm tra khái niệm, định nghĩa, cú pháp hoặc mục đích sử dụng đơn giản. "
                "Câu trả lời ngắn (1-2 câu)."
            ),
            QuestionDifficulty.EASY: (
                "cơ bản – yêu cầu giải thích khái niệm hoặc ví dụ đơn giản minh họa cách hoạt động. "
                "Có thể bao gồm một đoạn code ngắn (dưới 10 dòng) để thí sinh phân tích."
            ),
            QuestionDifficulty.MEDIUM: (
                "trung cấp – ứng dụng thực tế, kết hợp 1–2 khái niệm chính trong cùng một ví dụ. "
                "Câu trả lời yêu cầu phân tích ngắn hoặc trình bày code tầm 15–25 dòng."
                # "Có thể đưa ra 1 đoa code và cho thí sinh phân tích"
            ),
            QuestionDifficulty.HARD: (
                "nâng cao – yêu cầu phân tích sâu hoặc thiết kế mô-đun nhỏ (tối đa 3 lớp), "
                "kết hợp 2–3 nguyên lý hoặc khái niệm chính. "
                "Không yêu cầu xây dựng hệ thống hoàn chỉnh. "
                "Code minh họa nên nằm trong khoảng 30–50 dòng."
            ),
            QuestionDifficulty.VERY_HARD: (
                "rất khó – yêu cầu tổng hợp kiến thức hoặc mô phỏng một hệ thống hoàn chỉnh, "
                "áp dụng nhiều nguyên lý cùng lúc , có tính mở rộng và tái sử dụng. "
                "Độ dài code có thể vượt quá 60 dòng và mang tính thiết kế thực tế."
            )
        }

        history_text = memory.build_prompt()

        prompt = f"""
Bạn là một Interviewer AI THÔNG MINH và CÓ NHẬN THỨC.

THÔNG TIN THÍ SINH:
{candidate_context}

LỊCH SỬ HỘI THOẠI:
{history_text}

Dựa trên:
1. Background của thí sinh
2. Cách thí sinh trả lời các câu trước
3. Điểm mạnh/yếu đã thể hiện

Hãy tạo câu hỏi về "{topic}" với độ khó: {difficulty_descriptions[difficulty]}


TÀI LIỆU THAM KHẢO:
{knowledge_text if knowledge_text else "Không có tài liệu"}
Tuy nhiên, hãy lưu ý bản tóm ắt tài liệu tham khảo này:
{outline_summary or "Không có đánh giá outline"}

→ Cố gắng không hỏi quá sâu vào những phần mà tài liệu bị đánh giá là 'thiếu'.

YÊU CẦU:
- Câu hỏi CÁ NHÂN HÓA, phù hợp với level của thí sinh
- Với những câu hỏi dạng lý thuyết/ khái niệm , chỉ đưa ra câu hỏi khi chắc chắn tìm được câu trả lời trong tài liệu
- Nếu thí sinh yếu ở điểm nào, có thể hỏi lại theo cách khác
- Nếu thí sinh giỏi, đẩy khó hơn một chút
- Tham khảo lịch sử để tránh lặp lại câu hỏi tương tự
- Nếu thí sinh trả lời tốt câu trước, dành 1 lời khen trước câu hỏi mới.
- Dung lượng hợp lý, không quá dài vì thời gian phỏng vấn có hạn:

OUTPUT: JSON
{{"question": " lời khen (nếu có) + câu hỏi cá nhân hóa..."}}
Các ví dụ code bạn hãy đă nó trong thẻ <pre><code class='language-java'>...</code></pre> để thuận tiện cho mình render về sau
        Dùng ký tự xuống dòng <br> để format câu hỏi cho dễ đọc, ví dụ code thì dùng ký tự xuống dòng như trong ngôn ngữ lập trình'.
        
"""
        print("Tạo ra câu hỏi với độ kh ó:", difficulty.name)
        result = self.llm.invoke(prompt)
        parsed = _clean_and_parse_json_response(result, ["question"])

        # Format question
        import re
        question = re.sub(
            r"<pre><code([^>]*)>([\s\S]*?)</code></pre>",
            lambda m: f"<pre><code{m.group(1)}>{m.group(2).replace('<br>', '\n')}</code></pre>",
            parsed['question']
        )
        print(f"Final generated question with context: {question}")

        return question
# =======================
# 5. Answer Evaluator
# =======================

class AnswerEvaluator:
    """Component chuyên chấm điểm câu trả lời"""

    def __init__(self, llm: GoogleGenerativeAI):
        self.llm = llm

    def evaluate(
            self,
            question: str,
            answer: str,
            knowledge_text: str,
            memory: ConversationMemory,
            topic: str = None,
            difficulty: str = None,
            phase: str = "technical"
    ) -> tuple[float, str]:
        """
        Đánh giá câu trả lời theo 3 lớp logic:
        1️⃣ So khớp nội dung câu trả lời với tài liệu (grounding).
        2️⃣ Kiểm tra các phần đúng dù không có trong tài liệu (reasoning mềm).
        3️⃣ Tổng hợp điểm và phân tích.
        """

        history_text = memory.build_prompt()

        prompt = f"""
        Bạn là giám khảo phỏng vấn Java, chấm điểm thí sinh dựa trên cả độ chính xác kỹ thuật 
        và mức độ phù hợp với tài liệu tham khảo.

        BỐI CẢNH:
        - Giai đoạn phỏng vấn: {phase}
        - Chủ đề: {topic or "Không xác định"}
        - Mức độ khó: {difficulty or "Không xác định"}

        --- LỊCH SỬ HỘI THOẠI ---
        {history_text}

        --- CÂU HỎI ---
        {question}

        --- CÂU TRẢ LỜI ---
        {answer}

        --- TÀI LIỆU THAM KHẢO ---
        {knowledge_text if knowledge_text else "Không có tài liệu"}

        HÃY THỰC HIỆN 3 BƯỚC:

        1️⃣ **Phân tích ý chính của câu trả lời:**
           - Liệt kê ngắn gọn các ý chính mà thí sinh đã nêu.

        2️⃣ **Đối chiếu từng ý với tài liệu:**
           - Nếu ý có trong tài liệu hoặc trùng khớp về nội dung → ✅ "Khớp"
           - Nếu ý KHÔNG có trong tài liệu nhưng đúng và hợp lý theo kiến thức chung → ⚙️ "Đúng ngoài tài liệu"
           - Nếu ý sai rõ ràng hoặc mâu thuẫn với tài liệu → ❌ "Sai"

        3️⃣ **Tổng hợp điểm và nhận xét:**
           - ✅ Khớp nhiều: 8–10 điểm
           - ⚙️ Đúng ngoài tài liệu: 6–8 điểm
           - ❌ Sai hoặc lệch: 0–4 điểm
           - Nếu tài liệu không đủ thông tin để đánh giá: 5 điểm, ghi "Tài liệu không đủ".

        TRẢ VỀ DỮ LIỆU DƯỚI DẠNG JSON:
        {{
          "score": <số nguyên hoặc float từ 0-10>,
          "analysis": "<phân tích ngắn gọn về các ý chính, chỉ ra phần nào khớp và phần nào không>"
        }}
        """

        try:
            result = self.llm.invoke(prompt)
            parsed = _parse_evaluation_response(result)
            # print("Raw evaluation response:", result)
            # print("Parsed evaluation:", parsed)

            score = float(parsed.get("score", 5.0))
            analysis = parsed.get("analysis", "Không có nhận xét")

            print(f"📊 Chấm điểm: {score}, Nhận xét: {analysis}")

            # Cập nhật vào memory
            memory.add("student", answer)
            memory.add("interviewer", f"📊 Điểm: {score}/10 - {analysis}")

            return score, analysis

        except Exception as e:
            print(f"Lỗi khi chấm điểm: {e}")
            return 5.0, "Lỗi khi chấm điểm, mặc định 5/10"


# =======================
# 6. Difficulty Adapter
# =======================

class DifficultyAdapter:
    """Component điều chỉnh độ khó"""

    def __init__(self, config: InterviewConfig):
        self.config = config

    def decide_next_action(self, score: float) -> str:
        """Quyết định action tiếp theo"""
        if score >= self.config.threshold_high:
            return "harder"
        elif score >= self.config.threshold_low:
            return "same"
        else:
            return "easier"

    def get_next_difficulty(
            self,
            current: QuestionDifficulty,
            action: str
    ) -> QuestionDifficulty:
        """Tính độ khó tiếp theo"""
        difficulties = list(QuestionDifficulty)
        current_idx = difficulties.index(current)

        if action == "harder" and current_idx < len(difficulties) - 1:
            return difficulties[current_idx + 1]
        elif action == "easier" and current_idx > 0:
            return difficulties[current_idx - 1]
        else:
            return current


# =======================
# 7. Session Manager
# =======================

class SessionManager:
    """Component quản lý sessions và persistence"""

    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client["interviewer_ai"]
        self.collection = self.db["interview_sessions"]
        self.sessions: Dict[str, InterviewState] = {}

    def create_session(self, state: InterviewState) -> str:
        """Tạo session mới"""
        self.sessions[state.candidate_name] = state
        return state.candidate_name

    def get_session(self, candidate_name: str) -> Optional[InterviewState]:
        """Lấy session"""
        return self.sessions.get(candidate_name)

    def save_session(self, candidate_name: str) -> str:
        """Lưu session vào MongoDB"""
        state = self.sessions.get(candidate_name)
        if not state:
            raise ValueError(f"Session not found: {candidate_name}")

        # Bỏ knowledge_text vì quá dài
        state.knowledge_text = ""

        # Convert state to dict
        state_dict = asdict(state)

        # Convert Enum fields
        state_dict['level'] = state.level.value
        state_dict['current_difficulty'] = state.current_difficulty.value

        for attempt in state_dict.get('history', []):
            if isinstance(attempt.get('difficulty'), Enum):
                attempt['difficulty'] = attempt['difficulty'].value

        # Tạo document để lưu
        session_data = {
            "state": state_dict,
            "timestamp": datetime.datetime.utcnow(),
            "status": "active" if not state.is_finished else "completed"
        }

        # Fix Enum cho current_phase nếu có
        if isinstance(session_data["state"].get("current_phase"), Enum):
            session_data["state"]["current_phase"] = session_data["state"]["current_phase"].value

        # Thực hiện lưu vào MongoDB
        result = self.collection.insert_one(session_data)
        return str(result.inserted_id)

    def resume_session(self, session_id: str) -> InterviewState:
        """Khôi phục session từ MongoDB"""
        session = self.collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            raise ValueError(f"Session not found in DB: {session_id}")

        state_dict = session["state"]
        # Reconstruct enums
        state_dict["level"] = Level(state_dict["level"])
        state_dict["current_difficulty"] = QuestionDifficulty(state_dict["current_difficulty"])

        # Reconstruct history
        history = []
        for attempt in state_dict["history"]:
            attempt["difficulty"] = QuestionDifficulty(attempt["difficulty"])
            history.append(QuestionAttempt(**attempt))
        state_dict["history"] = history

        state = InterviewState(**state_dict)
        self.sessions[state.candidate_name] = state

        return state

    def delete_session(self, candidate_name: str):
        """Xóa session"""
        self.sessions.pop(candidate_name, None)


# =======================
# 8. Knowledge Builder
# =======================

class KnowledgeBuilder:
    """Component xây dựng knowledge context"""

    def __init__(self, knowledge_db: Optional[FAISS] = None):
        self.knowledge_db = knowledge_db
        self.retriever = None
        if knowledge_db:
            self.retriever = knowledge_db.as_retriever(search_kwargs={"k": 5})

    @property
    def knowledge_db(self):
        return self._knowledge_db

    @knowledge_db.setter
    def knowledge_db(self, value):
        self._knowledge_db = value
        if value:
            self.retriever = value.as_retriever(search_kwargs={"k": 5})
            print("🔄 retriever auto-updated from new knowledge_db")
        else:
            self.retriever = None
            print("⚠️ retriever cleared (knowledge_db=None)")
    def build_context(
            self,
            topic: str,
            outline: Optional[List[str]] = None
    ) -> str:
        """Xây dựng knowledge context"""
        results = []

        if outline and len(outline) > 0:
            for item in outline:
                query = f"{topic} {item}"
                docs = self.retriever.invoke(query)
                results.extend(docs)
        else:
            docs = self.retriever.invoke(topic)
            results.extend(docs)

        # Loại trùng lặp
        seen = set()
        unique_docs = []
        for doc in results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return "\n\n".join([doc.page_content for doc in unique_docs])


# =======================
# 9. Adaptive Interviewer (Orchestrator)
# =======================

class AdaptiveInterviewer:
    """Main orchestrator - phối hợp các components"""

    def __init__(
            self,

            api_key: str = None,
            config: InterviewConfig = None,
            embeddings_model: str = "intfloat/multilingual-e5-large-instruct",
            llm_model: str = "gemini-2.5-flash",
            mongo_uri: str = "mongodb://localhost:27017/",
        device: str = "cpu"  # ✅ THÊM PARAMETER MỚI
    ):
        # Load API key
        self.api_key = api_key or loadapi()
        self.config = config or InterviewConfig()

        # ✅ Initialize embeddings with device control
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embeddings_model,
            model_kwargs={'device': device},
            encode_kwargs={'device': device}
        )
        # Load vector stores
        # self.cv_db = FAISS.load_local(
        #     cv_vectorstore_path,
        #     self.embeddings,
        #     allow_dangerous_deserialization=True
        # )
        self.cv_db=None
        # knowledge_db = FAISS.load_local(
        #     knowledge_vectorstore_path,
        #     self.embeddings,
        #     allow_dangerous_deserialization=True
        # )
        knowledge_db=None

        # Initialize LLM
        self.llm = GoogleGenerativeAI(
            model=llm_model,
            google_api_key=self.api_key,
            temperature=self.config.llm_temperature
        )

        # Initialize components
        self.question_generator = QuestionGenerator(self.llm, self.config)
        self.answer_evaluator = AnswerEvaluator(self.llm)
        self.difficulty_adapter = DifficultyAdapter(self.config)
        self.session_manager = SessionManager(mongo_uri)
        self.knowledge_builder = KnowledgeBuilder(knowledge_db)

        # Memory per session
        self.memories: Dict[str, ConversationMemory] = {}
        # ✅ NEW: Initialize WarmupManager
        self.warmup_manager = WarmupManager(self.llm)

    def _get_or_create_memory(self, candidate_name: str) -> ConversationMemory:
        """Lấy hoặc tạo memory cho candidate"""
        if candidate_name not in self.memories:
            self.memories[candidate_name] = ConversationMemory(
                max_turns=self.config.max_memory_turns
            )
        return self.memories[candidate_name]

    def load_candidate_profile(self, candidate_name: str) -> tuple[str, Level]:
        """Load hồ sơ và phân loại level"""
        profile_docs = self.cv_db.similarity_search(candidate_name, k=1)
        if not profile_docs:
            raise ValueError(f"Không tìm thấy hồ sơ cho {candidate_name}")

        profile_content = profile_docs[0].page_content
        print("Loaded profile content:", profile_content)

        score_match = re.search(r'Điểm 40%[:\s]+([0-9.]+)', profile_content)
        if score_match:
            score_40 = float(score_match.group(1))
            level = classify_level_from_score(score_40)
        else:
            level = Level.TRUNG_BINH  # Default

        return profile_content, level

    def start_interview(
            self,
            candidate_name: str,
            topic: str,
            outline: Optional[List[str]] = None,
            knowledge_text: Optional[str] = None,
            outline_summary=""
    ) -> Dict:
        """Bắt đầu phỏng vấn từ giai đoạn WARMUP"""

        # Load profile
        profile, level = self.load_candidate_profile(candidate_name)
        initial_difficulty = get_initial_difficulty(level, self.config)

        # ✅ Extract candidate context
        candidate_context = self.warmup_manager.extract_candidate_context(profile)

        # Build knowledge nếu chưa có
        if not knowledge_text:
            knowledge_text = self.knowledge_builder.build_context(topic, outline)

        # ✅ Create state với WARMUP phase
        state = InterviewState(
            candidate_name=candidate_name,
            profile=profile,
            level=level,
            topic=topic,
            current_difficulty=initial_difficulty,
            attempts_at_current_level=0,
            max_attempts_per_level=self.config.max_attempts_per_level,
            total_questions_asked=0,
            max_total_questions=self.config.max_total_questions,
            upper_level_reached=0,
            history=[],
            is_finished=False,
            outline=outline,
            knowledge_text=knowledge_text,
            current_phase=InterviewPhase.WARMUP,  # ✅ Start with warmup
            warmup_questions_asked=0,
            candidate_context=candidate_context , # ✅ Save context
            outline_summary=outline_summary
        )

        self.session_manager.create_session(state)
        memory = self._get_or_create_memory(candidate_name)

        # ✅ Generate WARMUP question
        question = self.warmup_manager.generate_warmup_question(
            candidate_name=candidate_name.split(',')[0],  # Lấy tên
            candidate_context=candidate_context,
            topic=topic,
            warmup_count=0
        )

        # Add to history (không chấm điểm)
        state.history.append(QuestionAttempt(
            question=question,
            answer="",
            score=0.0,  # Warmup không có điểm
            analysis="(warmup - không chấm điểm)",
            difficulty=QuestionDifficulty.VERY_EASY,
            timestamp=datetime.datetime.now().isoformat(),
            question_hash=calculate_question_hash(question)
        ))

        return {
            "candidate": candidate_name,
            "topic": topic,
            "profile": profile,
            "level": level.value,
            "question": question,
            "difficulty": "warmup",  # ✅ Đánh dấu là warmup
            "phase": "warmup"
        }

    def submit_answer(self, candidate_name: str, answer: str) -> Dict:
        """Submit câu trả lời - xử lý theo phase"""

        state = self.session_manager.get_session(candidate_name)
        if not state:
            return {"error": "Interview not started"}

        if not state.history:
            return {"error": "No question found"}

        last_attempt = state.history[-1]
        memory = self._get_or_create_memory(candidate_name)

        # ✅ Xử lý theo phase
        if state.current_phase == InterviewPhase.WARMUP:
            return self._handle_warmup_answer(state, answer, memory)
        elif state.current_phase == InterviewPhase.TECHNICAL:
            return self._handle_technical_answer(state, answer, memory)
        else:  # CLOSING
            return self._handle_closing(state)

    def _handle_warmup_answer(
            self,
            state: InterviewState,
            answer: str,
            memory: ConversationMemory
    ) -> Dict:
        """Xử lý câu trả lời warmup (KHÔNG chấm điểm)"""

        last_attempt = state.history[-1]

        # Lưu câu trả lời (không chấm điểm)
        last_attempt.answer = answer
        last_attempt.score = 0.0
        last_attempt.analysis = "✅ Cảm ơn bạn đã chia sẻ!"

        # Lưu vào memory
        memory.add("student", answer)
        memory.add("interviewer", "Cảm ơn bạn! ")

        state.warmup_questions_asked += 1

        # ✅ Check xem đã đủ warmup chưa
        if state.warmup_questions_asked >= state.max_warmup_questions:
            # Chuyển sang TECHNICAL phase
            state.current_phase = InterviewPhase.TECHNICAL

            # Generate câu hỏi technical đầu tiên
            next_question = self.question_generator.generate_with_context(
                state.topic,
                state.current_difficulty,
                state.knowledge_text,
                memory,
                state.candidate_context,  # ✅ Pass context


                "Bắt đầu phần chuyên môn",
                outline_summary=state.outline_summary,
            )

            state.history.append(QuestionAttempt(
                question=next_question,
                answer="",
                score=0.0,
                analysis="(pending)",
                difficulty=state.current_difficulty,
                timestamp=datetime.datetime.now().isoformat(),
                question_hash=calculate_question_hash(next_question)
            ))

            return {
                "finished": False,
                "score": 0,
                "analysis": "✅ Phần làm quen hoàn tất! Bây giờ chúng ta bắt đầu phần chuyên môn nhé.",
                "next_question": next_question,
                "difficulty": state.current_difficulty.value,
                "phase": "technical"  # ✅ Chuyển phase
            }
        else:
            # Generate câu warmup tiếp theo
            next_warmup = self.warmup_manager.generate_warmup_question(
                candidate_name=state.candidate_name.split(',')[0],
                candidate_context=state.candidate_context,
                topic=state.topic,
                warmup_count=state.warmup_questions_asked
            )

            state.history.append(QuestionAttempt(
                question=next_warmup,
                answer="",
                score=0.0,
                analysis="(warmup)",
                difficulty=QuestionDifficulty.VERY_EASY,
                timestamp=datetime.datetime.now().isoformat(),
                question_hash=calculate_question_hash(next_warmup)
            ))

            return {
                "finished": False,
                "score": 0,
                "analysis": "✅ Tuyệt vời!",
                "next_question": next_warmup,
                "difficulty": "warmup",
                "phase": "warmup"
            }

    def _handle_technical_answer(
            self,
            state: InterviewState,
            answer: str,
            memory: ConversationMemory
    ) -> Dict:
        """Xử lý câu trả lời technical (CÓ chấm điểm) - GIỐNG CODE CŨ"""

        last_attempt = state.history[-1]

        # Evaluate answer
        score, analysis = self.answer_evaluator.evaluate(
            last_attempt.question,
            answer,
            state.knowledge_text,
            memory
        )
        # print(f"Evaluated score: {score}, analysis: {analysis}")

        # Update attempt
        last_attempt.answer = answer
        last_attempt.score = score
        last_attempt.analysis = analysis

        # Update state
        self._update_state(state, score)

        # Check if finished
        if state.is_finished:
            summary = self._generate_summary(state)
            self.session_manager.save_session(state.candidate_name)

            return {"finished": True, "summary": summary}

        # Generate next technical question với context
        next_question = self.question_generator.generate_with_context(
            state.topic,
            state.current_difficulty,
            state.knowledge_text,
            memory,
            state.candidate_context,  # ✅ Context-aware
            f"Đã hỏi {state.total_questions_asked} câu",
            outline_summary=state.outline_summary,
        )

        state.history.append(QuestionAttempt(
            question=next_question,
            answer="",
            score=0.0,
            analysis="(pending)",
            difficulty=state.current_difficulty,
            timestamp=datetime.datetime.now().isoformat(),
            question_hash=calculate_question_hash(next_question)
        ))

        return {
            "finished": False,
            "score": score,
            "analysis": analysis,
            "next_question": next_question,
            "difficulty": state.current_difficulty.value,
            "phase": "technical"
        }

    def _handle_closing(self, state: InterviewState) -> Dict:
        """Xử lý kết thúc phỏng vấn"""
        summary = self._generate_summary(state)
        self.session_manager.save_session(state.candidate_name)
        return {"finished": True, "summary": summary}

    def _update_state(self, state: InterviewState, score: float):
        """Update state sau mỗi câu hỏi"""
        state.total_questions_asked += 1

        action = self.difficulty_adapter.decide_next_action(score)

        if action == "harder":
            state.upper_level_reached += 1
            if state.upper_level_reached <= self.config.max_upper_level:
                state.current_difficulty = self.difficulty_adapter.get_next_difficulty(
                    state.current_difficulty, "harder"
                )
                state.attempts_at_current_level = 0
            else:
                state.is_finished = True

        elif action == "same":
            state.attempts_at_current_level += 1

        else:  # easier
            state.current_difficulty = self.difficulty_adapter.get_next_difficulty(
                state.current_difficulty, "easier"
            )
            state.attempts_at_current_level += 1
            state.upper_level_reached = max(0, state.upper_level_reached - 1)

        # Check termination
        if (state.attempts_at_current_level >= state.max_attempts_per_level or
                state.total_questions_asked >= state.max_total_questions or
                state.is_finished):
            state.is_finished = True
            scores = [a.score for a in state.history if a.score > 0]
            state.final_score = sum(scores) / len(scores) if scores else 0.0

    def _generate_summary(self, state: InterviewState) -> Dict:
        """Generate final summary"""
        return {
            "candidate_info": {
                "name": state.candidate_name,
                "profile": state.profile,
                "classified_level": state.level.value
            },
            "interview_stats": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_questions": len(state.history),
                "final_score": state.final_score,
                "topic": state.topic,
                "outline": state.outline
            },
            "question_history": [
                {
                    "question_number": i,
                    "difficulty": attempt.difficulty.value,
                    "question": attempt.question,
                    "answer": attempt.answer,
                    "score": attempt.score,
                    "analysis": attempt.analysis
                }
                for i, attempt in enumerate(state.history, 1)
            ]
        }


# =======================
# 10. Usage Example
# =======================

if __name__ == "__main__":
    # Custom config
    custom_config = InterviewConfig(
        threshold_high=7.5,
        threshold_low=4.5,
        max_total_questions=10,
        llm_temperature=0.6
    )

    # Initialize interviewer
    interviewer = AdaptiveInterviewer(
        cv_vectorstore_path="NotUse/vector_db_csv",
        knowledge_vectorstore_path="NotUse/vector_db2chunk_nltk",
        config=custom_config
    )

    # Start interview
    result = interviewer.start_interview(
        candidate_name="Ngô Văn Phát,KT1",
        topic="Kiểu dữ liệu trong Java",
        outline=["Kiểu dữ lệu cơ sở", "Kiểu dữ liệu gói", "Chuỗi ký tự String"]
    )

    print("Started interview:", result)