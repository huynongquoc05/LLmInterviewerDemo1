# AdaptiveInterviewer: AI Interviewer với State Machine thông minh
import datetime

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from GetApikey import loadapi

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


# =======================
# 1. Enums & Data Classes
# =======================

class Level(Enum):
    YEU = "yeu"  # <5
    TRUNG_BINH = "trung_binh"  # 5-6.5
    KHA = "kha"  # 6.5-8
    GIOI = "gioi"  # 8-9
    XUAT_SAC = "xuat_sac"  # 9-10


class QuestionDifficulty(Enum):
    VERY_EASY = "very_easy"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class QuestionAttempt:
    question: str
    answer: str
    score: float
    analysis: str
    difficulty: QuestionDifficulty
    timestamp: str


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
    final_score: Optional[float] = None


# =======================
# 2. Configuration & Thresholds
# =======================

class InterviewConfig:
    # Thresholds để quyết định next step
    THRESHOLD_HIGH = 7.0  # >= 7: chuyển lên khó hơn
    THRESHOLD_LOW = 4.0  # < 4: giảm xuống dễ hơn

    # Limits
    MAX_ATTEMPTS_PER_LEVEL = 2
    MAX_TOTAL_QUESTIONS = 8
    MAX_UPPER_LEVEL = 2  # max level có thể tăng lên từ ban đầu

    # Difficulty progression mapping
    DIFFICULTY_MAP = {
        Level.YEU: [QuestionDifficulty.VERY_EASY, QuestionDifficulty.EASY],
        Level.TRUNG_BINH: [QuestionDifficulty.EASY, QuestionDifficulty.EASY],
        Level.KHA: [QuestionDifficulty.MEDIUM, QuestionDifficulty.HARD],
        Level.GIOI: [QuestionDifficulty.HARD, QuestionDifficulty.VERY_HARD],
        Level.XUAT_SAC: [QuestionDifficulty.VERY_HARD]
    }


# =======================
# 3. Utility Functions
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


def get_initial_difficulty(level: Level) -> QuestionDifficulty:
    """Lấy độ khó ban đầu cho level"""
    return InterviewConfig.DIFFICULTY_MAP[level][0]


def get_next_difficulty(current: QuestionDifficulty, action: str) -> QuestionDifficulty:
    """Tính độ khó tiếp theo dựa trên action (harder/same/easier)"""
    difficulties = list(QuestionDifficulty)
    current_idx = difficulties.index(current)

    if action == "harder" and current_idx < len(difficulties) - 1:
        return difficulties[current_idx + 1]
    elif action == "easier" and current_idx > 0:
        return difficulties[current_idx - 1]
    else:  # same or can't change
        return current


import json, re

def _clean_and_parse_json_response(raw_text: str, expected_keys: list[str] = None) -> dict:
    """Parse JSON từ LLM, xử lý cả khi trong string có code block markdown."""
    if not raw_text:
        return {}

    text = raw_text.strip()

    # Bóc ra phần giữa { ... }
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return {}

    candidate = text[first_brace:last_brace + 1]

    # Loại bỏ code fences kiểu ```java ... ```
    candidate = re.sub(r"```[a-zA-Z]*", "", candidate)  # bỏ ```java
    candidate = candidate.replace("```", "")            # bỏ ```

    # Chuẩn hóa xuống dòng trong chuỗi thành \n
    def _escape_newlines_in_strings(match):
        inner = match.group(0)
        return inner.replace("\n", "\\n")
    candidate = re.sub(r'\".*?\"', _escape_newlines_in_strings, candidate, flags=re.S)

    try:
        parsed = json.loads(candidate)
        if expected_keys:
            parsed = {k: v for k, v in parsed.items() if k in expected_keys}
        return parsed
    except Exception as e:
        print(f"⚠️ JSON parse error after cleaning: {e}")
        return {}



import re, json

def _clean_and_parse_single_question(raw_text: str) -> str:
    """
    Input: raw_text từ LLM (có thể kèm ```json``` hoặc lộn xộn)
    Output: 1 string câu hỏi sạch hoặc "" nếu không parse được
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # 1) Nếu có code fence ```json ... ```
    code_fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if code_fence_match:
        text = code_fence_match.group(1).strip()

    # 2) Thử parse JSON object
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = text[first_brace:last_brace+1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "question" in parsed:
                return _sanitize_question(parsed["question"])
        except Exception:
            pass

    # 3) Nếu thất bại, thử tìm chuỗi trong ngoặc kép
    quoted = re.findall(r'"([^"]{10,})"', text, flags=re.S)  # chuỗi dài ≥10 ký tự
    if quoted:
        return _sanitize_question(quoted[0])

    # 4) Fallback: lấy dòng dài nhất làm câu hỏi
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 20]
    if lines:
        return _sanitize_question(max(lines, key=len))

    return ""


def _sanitize_question(q: str) -> str:
    """Làm sạch 1 câu hỏi: bỏ backticks, quotes, số thứ tự..."""
    s = str(q).strip()
    s = re.sub(r'^[`\"]+|[`\"]+$', '', s).strip()
    s = re.sub(r'^\s*"\s*', '', s)
    s = re.sub(r'^\s*\(?\d+\)?[\).\s:-]+\s*', '', s)
    s = s.rstrip(",;}]")
    return s.strip()


# =======================
# 4. Core Interviewer Class
# =======================

class AdaptiveInterviewer:
    def __init__(self):
        # Load components
        self.api_key = loadapi()
        self.embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large-instruct")
        self.cv_db = FAISS.load_local("vector_db_csv", self.embeddings, allow_dangerous_deserialization=True)
        self.knowledge_db = FAISS.load_local("vector_db2chunk_nltk", self.embeddings,
                                             allow_dangerous_deserialization=True)
        self.retriever = self.knowledge_db.as_retriever(search_kwargs={"k": 5})
        self.llm = GoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=self.api_key,
            temperature=0.7
        )
        # === New: conversation memory (simple list) ===
        self.memory: list[dict] = []
        self.max_memory_turns = 6   # chỉ giữ 6 lượt gần nhất

    # ============ Memory Helpers ============
    def add_to_memory(self, role: str, content: str):
        """Thêm một đoạn hội thoại vào memory."""
        self.memory.append({"role": role, "content": content})
        self.memory = self.memory[-self.max_memory_turns:]  # cắt bớt nếu quá dài

    def build_history_prompt(self) -> str:
        """Ghép memory thành đoạn hội thoại để truyền vào LLM."""
        if not self.memory:
            return ""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.memory])

    def load_candidate_profile(self, candidate_name: str) -> tuple[str, Level]:
        """Load hồ sơ và phân loại level"""
        profile_docs = self.cv_db.similarity_search(candidate_name, k=1)
        if not profile_docs:
            raise ValueError(f"Không tìm thấy hồ sơ cho {candidate_name}")

        profile_content = profile_docs[0].page_content

        # Extract điểm 40% từ profile (giả sử có format chuẩn)
        score_match = re.search(r'Điểm 40%[:\s]+([0-9.]+)', profile_content)
        if score_match:
            score_40 = float(score_match.group(1))
            level = classify_level_from_score(score_40)
        else:
            # Fallback: dùng LLM để classify
            level = self._classify_level_with_llm(profile_content)

        return profile_content, level

    def _classify_level_with_llm(self, profile: str) -> Level:
        """Fallback method để classify level bằng LLM"""
        classify_prompt = f"""
        Bạn là một Interviewer AI và đang chuẩn bị phỏng vấn bài thi vấn đáp của 1 thí sinh .
        Phân loại trình độ thí sinh theo điểm 40%: Yếu (<5), Trung bình (5-6.5), Khá (6.5-8), Giỏi (8-9), Xuất sắc (9-10).

        Hồ sơ: {profile}

        Trả về JSON: {{"level": "yeu|trung_binh|kha|gioi|xuat_sac"}}
        """
        result = self.llm.invoke(classify_prompt)
        parsed = _clean_and_parse_json_response(result, ["level"])
        level_str = parsed.get("level", "trung_binh")

        # Convert to enum
        level_mapping = {
            "yeu": Level.YEU,
            "trung_binh": Level.TRUNG_BINH,
            "kha": Level.KHA,
            "gioi": Level.GIOI,
            "xuat_sac": Level.XUAT_SAC
        }
        return level_mapping.get(level_str, Level.TRUNG_BINH)

    def generate_question(self, topic: str, difficulty: QuestionDifficulty, context: str = "") -> str:
        """Generate câu hỏi theo topic và độ khó"""
        knowledge_context = self.retriever.invoke(f"{topic} {difficulty.value}")
        knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_context])
        history_text = self.build_history_prompt()  # Lấy lịch sử hội thoại
        difficulty_descriptions = {
            QuestionDifficulty.VERY_EASY: "rất cơ bản, định nghĩa đơn giản",
            QuestionDifficulty.EASY: "cơ bản, ví dụ thực tế",
            QuestionDifficulty.MEDIUM: "trung cấp, ứng dụng thực tế",
            QuestionDifficulty.HARD: "nâng cao, phân tích sâu",
            QuestionDifficulty.VERY_HARD: "rất khó, tổng hợp kiến thức"
        }

        generate_prompt = f"""
        Bạn là một Interviewer AI.
         Đây là lịch sử hội thoại gần đây, :
        {history_text}
        Tạo 1 câu hỏi phỏng vấn Java về chủ đề "{topic}" với độ khó "{difficulty_descriptions[difficulty]}".

        {context if context else ""}

        Tài liệu tham khảo:
        {knowledge_text}

        Yêu cầu:
        - Câu hỏi rõ ràng, cụ thể, phải lấy từ tài liệu tham khảo, không hỏi lan man
        - Phù hợp độ khó yêu cầu
        - Tiếng Việt, hạn chế những cụm từ như "theo tài liệu tham khảo" trong câu hỏi

        Trả về **CHỈ** **một object JSON thuần** có dạng: {{"question": "câu hỏi..."}}
        - KHÔNG kèm lời chào, giải thích, hay code fence (```).
        """

        result = self.llm.invoke(generate_prompt)
        print(result)
        parsed = _clean_and_parse_json_response(result, ["question"])
        print(parsed)
        self.add_to_memory("interviewer", parsed.get("question", "Hãy giải thích về Java?"))
         # Thêm câu hỏi vào memory
        print(self.memory)
        return parsed.get("question", "Hãy giải thích về Java?")

    def evaluate_answer(self, question: str, answer: str, topic: str) -> tuple[float, str]:
        """Đánh giá câu trả lời và trả về (score, analysis)"""
        knowledge_context = self.retriever.invoke(topic)
        knowledge_text = "\n\n".join([doc.page_content for doc in knowledge_context])
        history_text = self.build_history_prompt()
        eval_prompt = f"""
        Đây là lịch sử hội thoại gần đây:
        {history_text}
        Chấm điểm câu trả lời phỏng vấn Java (0-10 điểm).

        Câu hỏi: {question}
        Câu trả lời: {answer}

        Tài liệu tham chiếu:
        {knowledge_text}

        Đánh giá về: tính chính xác, đầy đủ, rõ ràng.

        Trả về JSON: {{
            "score": <số từ 0-10>,
            "analysis": "<nhận xét ngắn gọn>"
        }}
        """
        print(history_text)
        result = self.llm.invoke(eval_prompt)
        parsed = _clean_and_parse_json_response(result, ["score", "analysis"])

        score = float(parsed.get("score", 5.0))
        analysis = parsed.get("analysis", "Không có nhận xét")
        # === Cập nhật memory ===
        self.add_to_memory("student", answer)
        self.add_to_memory("interviewer", f"📊 Điểm: {score}/10 - {analysis}")
        print("current memory:", self.memory)
        return score, analysis

    def decide_next_action(self, score: float, state: InterviewState) -> str:
        """Policy Engine: quyết định action tiếp theo"""
        if score >= InterviewConfig.THRESHOLD_HIGH:
            return "harder"
        elif score >= InterviewConfig.THRESHOLD_LOW:
            return "same"
        else:
            return "easier"

    def update_state_after_question(self, state: InterviewState,
                                    question: str, answer: str,
                                    score: float, analysis: str) -> None:
        """Update state sau mỗi câu hỏi"""
        # Add to history
        attempt = QuestionAttempt(
            question=question,
            answer=answer,
            score=score,
            analysis=analysis,
            difficulty=state.current_difficulty,
            timestamp=str(len(state.history) + 1)
        )
        state.history.append(attempt)
        state.total_questions_asked += 1

        # Decide next action
        action = self.decide_next_action(score, state)

        if action == "harder":
            state.upper_level_reached += 1
            if state.upper_level_reached <= InterviewConfig.MAX_UPPER_LEVEL:
                state.current_difficulty = get_next_difficulty(state.current_difficulty, "harder")
                state.attempts_at_current_level = 0

            else:
                # Đã vượt giới hạn nâng cấp
                state.is_finished = True
        elif action == "same":
            # Stay same level but generate different question
            state.attempts_at_current_level += 1
        else:  # easier
            # Move to easier, increase attempts
            state.current_difficulty = get_next_difficulty(state.current_difficulty, "easier")
            state.attempts_at_current_level += 1

        # Check termination conditions
        # Kiểm tra điều kiện kết thúc
        if (state.attempts_at_current_level >= InterviewConfig.MAX_ATTEMPTS_PER_LEVEL or
                state.total_questions_asked >= InterviewConfig.MAX_TOTAL_QUESTIONS or
                state.is_finished):
            state.is_finished = True
            scores = [attempt.score for attempt in state.history]
            state.final_score = sum(scores) / len(scores) if scores else 0.0

    def run_interview(self, candidate_name: str, topic: str) -> Dict:
        """Main interview loop"""
        print(f"🎯 Bắt đầu phỏng vấn: {candidate_name} - Chủ đề: {topic}")

        # 1. Load candidate profile & classify
        profile, level = self.load_candidate_profile(candidate_name)
        initial_difficulty = get_initial_difficulty(level)

        # 2. Initialize state
        state = InterviewState(
            candidate_name=candidate_name,
            profile=profile,
            level=level,
            topic=topic,
            current_difficulty=initial_difficulty,
            attempts_at_current_level=0,
            max_attempts_per_level=InterviewConfig.MAX_ATTEMPTS_PER_LEVEL,
            total_questions_asked=0,
            max_total_questions=InterviewConfig.MAX_TOTAL_QUESTIONS,
            history=[],
            is_finished=False,
            upper_level_reached=0
        )

        print(f"📋 Hồ sơ: {profile}")
        print(f"📊 Level: {level.value} - Độ khó ban đầu: {initial_difficulty.value}")
        print("\n" + "=" * 50)

        # 3. Main interview loop
        while not state.is_finished:
            try:
                # Generate question
                context_hint = f"Đã hỏi {state.total_questions_asked} câu. " \
                               f"Attempts ở level hiện tại: {state.attempts_at_current_level}"
                question = self.generate_question(topic, state.current_difficulty, context_hint)

                # Ask question
                print(f"\n🤖 Câu hỏi #{state.total_questions_asked + 1} (Độ khó: {state.current_difficulty.value}):")
                print(f"   {question}")

                # Get answer
                answer = input("👩‍🎓 Thí sinh trả lời: ").strip()

                # Evaluate
                score, analysis = self.evaluate_answer(question, answer, topic)
                print(f"📊 Điểm: {score}/10 - {analysis}")

                # Update state
                self.update_state_after_question(state, question, answer, score, analysis)

                # Show state info
                if not state.is_finished:
                    action = self.decide_next_action(score, state)
                    print(
                        f"🔄 Next action: {action} (Attempts: {state.attempts_at_current_level}/{InterviewConfig.MAX_ATTEMPTS_PER_LEVEL})")

            except KeyboardInterrupt:
                print("\n⏹️ Phỏng vấn bị dừng bởi người dùng")
                state.is_finished = True
                scores = [attempt.score for attempt in state.history]
                state.final_score = sum(scores) / len(scores) if scores else 0.0
                break
            except Exception as e:
                print(f"❌ Lỗi: {e}")
                # Continue with a simple fallback question
                continue

        # 4. Generate summary
        return self.generate_summary(state)

    def generate_summary(self, state: InterviewState) -> Dict:
        """Generate final interview summary"""
        print("\n" + "=" * 50)
        print("📝 TỔNG KẾT PHỎNG VẤN")
        print("=" * 50)

        summary = {
            "candidate_info": {
                "name": state.candidate_name,
                "profile": state.profile,
                "classified_level": state.level.value
            },
            "interview_stats": {
                "timestamp": datetime.datetime.now().isoformat(),
                "total_questions": len(state.history),
                "final_score": state.final_score,
                "topic": state.topic
            },
            "question_history": []
        }

        for i, attempt in enumerate(state.history, 1):
            q_info = {

                "question_number": i,
                "difficulty": attempt.difficulty.value,
                "question": attempt.question,
                "answer": attempt.answer,
                "score": attempt.score,
                "analysis": attempt.analysis
            }
            summary["question_history"].append(q_info)

            print(f"\nCâu {i} ({attempt.difficulty.value}):")
            print(f"Q: {attempt.question}")
            print(f"A: {attempt.answer}")
            print(f"Score: {attempt.score}/10 - {attempt.analysis}")

        print(f"\n🏆 ĐIỂM TỔNG KẾT: {state.final_score:.1f}/10")

        return summary


# =======================
# 5. Usage Example
# =======================

if __name__ == "__main__":
    from pymongo import MongoClient

    # Kết nối MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["interviewer_ai"]
    collection = db["interview_results"]
    interviewer = AdaptiveInterviewer()

    # Test cases
    test_cases = [
        ("Hoàng Thị Oanh,QTKD2", "Kiểu dữ liệu trong Java"),

    ]

    for candidate, topic in test_cases[:1]:  # Chỉ test 1 case đầu tiên
        result = interviewer.run_interview(candidate, topic)

        # Save results
        with open(f"InterviewScripts/interview_result_{candidate.replace(',', '_')}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        collection.insert_one(result)
         # Lưu vào MongoDB
        print(f"✅ Kết quả đã lưu vào file JSON")
        print("memory",interviewer.memory)
        break