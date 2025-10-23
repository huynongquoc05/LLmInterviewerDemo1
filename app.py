from flask import Flask, request, jsonify, render_template, send_file
from pymongo import MongoClient
# Import refactored interviewer
from LLMInterviewer4 import (
    AdaptiveInterviewer,
    InterviewConfig, KnowledgeBuilder
)
from gtts import gTTS
import os

import uuid
import threading
import time
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from flask import Response, stream_with_context

from extension import build_cv_vectorstore_from_candidates
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# Tạo custom config nếu cần
custom_config = InterviewConfig(
    threshold_high=7.0,
    threshold_low=4.0,
    max_attempts_per_level=2,
    max_total_questions=8,
    max_upper_level=2,
    llm_temperature=0.5,
    max_memory_turns=6
)
print("🔧 Đang khởi tạo AdaptiveInterviewer toàn cục...")

global_interviewer = AdaptiveInterviewer(

    config=custom_config,
    embeddings_model="intfloat/multilingual-e5-large-instruct",
    llm_model="gemini-2.5-flash",
    mongo_uri="mongodb://localhost:27017/"
)

print("✅ AdaptiveInterviewer đã được load (1 lần duy nhất).")

cv_vectorstore_path = "NotUse/vector_db_csv"  # CV vectorstore cố định
# Import từ BuildVectorStores.py (cải tiến)
from BuildVectorStores import (
    build_vectorstore,
    list_vectorstores,
    delete_vectorstore,
    VALID_MODELS
)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Thư mục lưu trữ audio tạm thời
AUDIO_FOLDER = "temp_audio"
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)

# Dictionary để lưu trữ thông tin audio files
audio_cache = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Kết nối MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["interviewer_ai"]
collection = db["interview_results"]

app = Flask(__name__, static_url_path='/iview1/static', static_folder='static')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# 🧠 Hàm xác định base_path tự động (khi render template)
@app.context_processor
def inject_base_path():
    """
    Nếu đang chạy trên domain ánh xạ fit.neu.edu.vn thì thêm prefix /iview1,
    ngược lại (chạy bằng IP hoặc localhost) thì để trống.
    """
    if 'fit.neu.edu.vn' in request.host:
        base_path = '/iview1'
    else:
        base_path = ''
    return dict(base_path=base_path)

@app.route("/")
def index():
    """Trang chủ - có thể tạo landing page""" #
    #Cleanup old audio files
    clean_old_audio_files()
    return render_template("home.html") # Tạo trang chủ riêng nếu cần

def clean_old_audio_files():
    """Xóa các file audio cũ hơn 1 giờ"""
    try:
        now = datetime.now()
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                if now - file_time > timedelta(hours=1):
                    os.remove(file_path)
                    print(f"Đã xóa file audio cũ: {filename}")
    except Exception as e:
        print(f"Lỗi khi xóa file audio cũ: {e}")

def cleanup_temp_files():
    """Xóa tất cả file tạm khi kết thúc chương trình"""
    try:
        for filename in os.listdir(AUDIO_FOLDER):
            file_path = os.path.join(AUDIO_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Đã xóa file tạm: {filename}")
    except Exception as e:
        print(f"Lỗi khi xóa file tạm: {e}")

import re
from bs4 import BeautifulSoup

def remove_code_blocks(text: str) -> str:
    """
    Loại bỏ các khối code HTML (<pre><code>...</code></pre>)
    và làm sạch các thẻ HTML khác để tạo nội dung gọn gàng cho TTS.
    """
    if not text:
        return ""

    # Bóc nội dung HTML
    soup = BeautifulSoup(text, "html.parser")

    # Xóa toàn bộ các thẻ code block
    for code_block in soup.find_all(["pre", "code"]):
        code_block.decompose()

    # Lấy phần text còn lại (đã bỏ HTML tags)
    clean_text = soup.get_text(separator=" ", strip=True)

    # Xử lý gọn các khoảng trắng, xuống dòng thừa
    clean_text = re.sub(r'\s+', ' ', clean_text)

    return clean_text.strip()




def create_audio_from_text(text, lang='vi'):
    """
    Tạo file audio từ text.
    Ưu tiên dùng ElevenLabs (tự nhiên hơn), fallback sang gTTS nếu lỗi hoặc hết hạn mức.

    Args:
        text (str): Văn bản cần chuyển thành giọng nói
        lang (str): Ngôn ngữ ('vi' cho tiếng Việt, 'en' cho tiếng Anh)

    Returns:
        str | None: ID của audio (uuid) nếu thành công, None nếu thất bại
    """
    from extension import generate_voice_ElevenLab  # Hàm tiện ích bạn đã có
    try:
        # Làm sạch text
        clean_text = remove_code_blocks(text) if 'remove_code_blocks' in globals() else text
        clean_text = clean_text.replace("🤖", "").strip()
        if not clean_text:
            print("⚠️ Text trống, bỏ qua tạo audio.")
            return None

        # Tạo tên file unique
        audio_id = str(uuid.uuid4())
        audio_filename = f"question_{audio_id}.mp3"
        audio_path = os.path.join(AUDIO_FOLDER, audio_filename)

        # ===== ƯU TIÊN ELEVENLABS =====
        print("🧠 Đang tạo voice bằng ElevenLabs...")
        eleven_audio_path = generate_voice_ElevenLab(clean_text, output_path=audio_path)

        if eleven_audio_path:
            # Lưu cache
            audio_cache[audio_id] = {
                'filename': audio_filename,
                'path': eleven_audio_path,
                'created_at': datetime.now(),
                'text': clean_text,
                'source': 'elevenlabs'
            }
            print(f"✅ Đã tạo audio bằng ElevenLabs: {audio_filename}")
            return audio_id

        # ===== FALLBACK: GOOGLE TTS =====
        print("⚠️ ElevenLabs lỗi hoặc hết hạn mức, fallback sang Google TTS...")
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        tts.save(audio_path)

        audio_cache[audio_id] = {
            'filename': audio_filename,
            'path': audio_path,
            'created_at': datetime.now(),
            'text': clean_text,
            'source': 'gtts'
        }

        print(f"✅ Đã tạo audio bằng Google TTS: {audio_filename}")
        return audio_id

    except Exception as e:
        print(f"❌ Lỗi tạo audio: {e}")
        return None


def detect_language(text):
    """
    Phát hiện ngôn ngữ của văn bản (đơn giản)
    """
    # Kiểm tra xem có ký tự tiếng Việt không
    vietnamese_chars = "àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵđ"
    vietnamese_chars += vietnamese_chars.upper()

    if any(char in text for char in vietnamese_chars):
        return 'vi'
    else:
        return 'en'




@app.route("/interviewing")
def interviewing():
    """Trang phỏng vấn chính"""
    clean_old_audio_files()
    return render_template("interviewing.html")


@app.route("/interviewing/vectorstores", methods=["GET"])
def get_available_vectorstores():
    """Lấy danh sách vectorstores để chọn"""
    try:
        vectorstores = list_vectorstores(mongo_uri="mongodb://localhost:27017/")

        # Format cho dropdown
        options = []
        for vs in vectorstores:
            options.append({
                "id": vs["_id"],
                "name": vs["vectorstore_name"],
                "path": vs["vectorstore_path"],
                "topic": vs.get("custom", {}).get("topic", "Unknown"),
                "pdf_file": vs["pdf_file"],
                "created_at": vs["created_at"],
                "num_chunks": vs["num_chunks"]
            })

        return jsonify({
            "success": True,
            "vectorstores": options
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



@app.route("/interviewing/answer", methods=["POST"])
def answer():
    data = request.json
    candidate, answer = data["candidate"], data["answer"]

    result = global_interviewer.submit_answer(candidate, answer)

    # Nếu phỏng vấn chưa kết thúc và có câu hỏi tiếp theo
    if not result.get("finished") and "next_question" in result:
        question_text = result["next_question"]
        lang = detect_language(question_text)

        # 🔹 Chỉ tạo audio nếu phase != closing
        if result.get("phase") != "closing":
            audio_id = create_audio_from_text(question_text, lang)
            if audio_id:
                result["audio_id"] = audio_id
                result["audio_url"] = f"/audio/{audio_id}"

    # Lưu kết quả vào MongoDB nếu phỏng vấn kết thúc
    if result.get("finished"):
        insert_result = collection.insert_one(result["summary"])
        result["summary"]["_id"] = str(insert_result.inserted_id)

    # 🔹 Đảm bảo FE biết phase hiện tại
    result.setdefault("phase", "technical")

    return jsonify(result)


@app.route("/audio/<audio_id>")
def serve_audio(audio_id):
    """
    Endpoint để phục vụ file audio
    """
    try:
        if audio_id in audio_cache:
            audio_path = audio_cache[audio_id]['path']
            if os.path.exists(audio_path):
                return send_file(
                    audio_path,
                    mimetype="audio/mpeg",
                    as_attachment=False,
                    download_name=f"question_{audio_id}.mp3"
                )

        return jsonify({"error": "Audio file not found"}), 404

    except Exception as e:
        print(f"Lỗi khi phục vụ audio: {e}")
        return jsonify({"error": "Error serving audio file"}), 500


@app.route("/audio/info/<audio_id>")
def audio_info(audio_id):
    """
    Endpoint để lấy thông tin về audio file
    """
    if audio_id in audio_cache:
        info = audio_cache[audio_id].copy()
        info['created_at'] = info['created_at'].isoformat()
        info.pop('path', None)  # Không trả về đường dẫn file
        return jsonify(info)

    return jsonify({"error": "Audio not found"}), 404


@app.route("/test-tts", methods=["POST"])
def test_tts():
    """
    Endpoint để test tính năng TTS
    """
    data = request.json
    text = data.get("text", "Xin chào, đây là bài test text-to-speech")
    lang = data.get("lang", "vi")

    audio_id = create_audio_from_text(text, lang)
    if audio_id:
        return jsonify({
            "success": True,
            "audio_id": audio_id,
            "audio_url": f"/audio/{audio_id}",
            "text": text,
            "language": lang
        })
    else:
        return jsonify({
            "success": False,
            "error": "Failed to create audio"
        }), 500


# Background task để dọn dẹp file audio định kỳ
def cleanup_scheduler():
    """
    Chạy cleanup mỗi 30 phút
    """
    while True:
        time.sleep(30 * 60)  # 30 phút
        clean_old_audio_files()

        # Dọn dẹp cache
        now = datetime.now()
        expired_keys = []
        for audio_id, info in audio_cache.items():
            if now - info['created_at'] > timedelta(hours=1):
                expired_keys.append(audio_id)

        for key in expired_keys:
            audio_cache.pop(key, None)

        if expired_keys:
            print(f"Đã xóa {len(expired_keys)} audio cache entries")


# ================================
# Embedding Routes
# ================================

@app.route("/embedding")
def embedding_page():
    """Trang quản lý vectorstore"""
    return render_template("embedding.html")


@app.route("/embedding/upload", methods=["POST"])
def upload_pdf():
    """
    Upload PDF và tạo vectorstore với Server-Sent Events để report progress
    """
    if 'pdf_file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['pdf_file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files allowed"}), 400

    # Get parameters
    chunk_size = int(request.form.get('chunk_size', 1600))
    chunk_overlap = int(request.form.get('chunk_overlap', 400))
    model_name = request.form.get('model_name', 'intfloat/multilingual-e5-large-instruct')
    splitter_strategy = request.form.get('splitter_strategy', 'nltk')

    # Validate model
    if model_name not in VALID_MODELS:
        return jsonify({"error": f"Invalid model: {model_name}"}), 400

    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)

    file.save(filepath)

    def generate():
        """Generator function cho SSE"""
        progress_data = {"stage": "init", "progress": 0}

        def progress_callback(stage, progress):
            progress_data['stage'] = stage
            progress_data['progress'] = progress

        try:
            # Send initial progress
            yield f"data: {json.dumps({'status': 'processing', 'stage': 'init', 'progress': 0})}\n\n"

            # Build vectorstore
            save_path, metadata = build_vectorstore(
                pdf_path=filepath,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                model_name=model_name,
                mongo_uri="mongodb://localhost:27017/",
                splitter_strategy=splitter_strategy,
                skip_duplicate=True,
                custom_metadata={
                    "uploaded_filename": filename,
                    "original_path": filepath
                },
                progress_callback=progress_callback
            )

            # Send progress updates
            yield f"data: {json.dumps({'status': 'processing', **progress_data})}\n\n"

            # Success
            yield f"data: {json.dumps({'status': 'completed', 'progress': 100, 'metadata': metadata})}\n\n"

        except Exception as e:
            # Error
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

        finally:
            # Optionally delete uploaded file after processing
            # os.remove(filepath)
            pass

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route("/embedding/list", methods=["GET"])
def list_vectorstores_route():
    """Lấy danh sách tất cả vectorstores"""
    try:
        vectorstores = list_vectorstores(mongo_uri="mongodb://localhost:27017/")
        return jsonify({
            "success": True,
            "vectorstores": vectorstores,
            "count": len(vectorstores)
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/embedding/delete/<vectorstore_id>", methods=["DELETE"])
def delete_vectorstore_route(vectorstore_id):
    """Xóa vectorstore"""
    try:
        success = delete_vectorstore(
            vectorstore_id=vectorstore_id,
            mongo_uri="mongodb://localhost:27017/",
            remove_files=True
        )

        if success:
            return jsonify({
                "success": True,
                "message": "Vectorstore deleted successfully"
            })
        else:
            return jsonify({
                "success": False,
                "message": "Vectorstore not found"
            }), 404

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route("/embedding/models", methods=["GET"])
def get_models():
    """Lấy danh sách models available"""
    return jsonify({
        "success": True,
        "models": VALID_MODELS
    })


@app.route("/embedding/info/<vectorstore_id>", methods=["GET"])
def get_vectorstore_info(vectorstore_id):
    """Lấy thông tin chi tiết của một vectorstore"""
    try:
        from bson import ObjectId
        from pymongo import MongoClient

        client = MongoClient("mongodb://localhost:27017/")
        db = client["interviewer_ai"]
        collection = db["vectorstores"]

        vs = collection.find_one({"_id": ObjectId(vectorstore_id)})

        if not vs:
            return jsonify({
                "success": False,
                "message": "Vectorstore not found"
            }), 404

        vs["_id"] = str(vs["_id"])

        return jsonify({
            "success": True,
            "vectorstore": vs
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# ================================
# Additional Routes (Optional)
# ================================


# ================================
# MAIN
# ================================
# USAGE: Thêm các routes này vào app.py hiện tại
# Đảm bảo import các thư viện cần thiết ở đầu file
# ================================

#Code mới
# ================================
# Interview Session Management Routes
# Thêm vào app.py sau các routes hiện tại
# ================================

from bson import ObjectId
from datetime import datetime
import json


# ================================
# SESSION MANAGEMENT
# ================================

@app.route("/interview_session")
def interview_session_page():
    """Trang quản lý buổi phỏng vấn"""
    return render_template("interview_session.html")


@app.route("/interview_session/create", methods=["POST"])
def create_interview_session():
    """
    Tạo buổi phỏng vấn mới với:
    - Config parameters
    - Danh sách thí sinh
    - Knowledge base (vectorstore_id)
    - Topic và outline
    - Pre-load knowledge_text từ vectorstore
    """
    try:
        data = request.json

        # Validate required fields
        required_fields = ['session_name', 'config', 'candidates', 'vectorstore_id', 'topic']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing field: {field}"}), 400

        # Get vectorstore info
        vs_collection = db["vectorstores"]
        vectorstore = vs_collection.find_one({"_id": ObjectId(data["vectorstore_id"])})

        if not vectorstore:
            return jsonify({"success": False, "error": "Vectorstore not found"}), 404

        knowledge_vectorstore_path = vectorstore["vectorstore_path"]
        # ✅ 1. Sinh vectorstore cho danh sách thí sinh
        cv_vectorstore_path = build_cv_vectorstore_from_candidates(data["candidates"])

        # Pre-load knowledge text
        from LLMInterviewer4 import KnowledgeBuilder


        # # Load embeddings
        # embeddings = HuggingFaceEmbeddings(
        #     model_name="intfloat/multilingual-e5-large-instruct"
        # )

        # Load knowledge vectorstore
        knowledge_db = FAISS.load_local(
            knowledge_vectorstore_path,
            global_interviewer.embeddings,
            allow_dangerous_deserialization=True
        )

        # Build knowledge context
        kb = global_interviewer.knowledge_builder
        kb.knowledge_db = knowledge_db


        # Build knowledge text
        knowledge_text = kb.build_context(
            topic=data["topic"],
            outline=data.get("outline", None)
        )
        from extension import summarize_knowledge_with_llm

        report = summarize_knowledge_with_llm(
            knowledge_text,
            topic=data["topic"],
            outline=data.get("outline", []),
            llm=global_interviewer.llm
        )

        # Create session document
        session_doc = {
            "session_name": data["session_name"],
            "config": data["config"],
            "candidates": data["candidates"],  # List of {name, class, status: 'pending'}
            "cv_vectorstore_path": cv_vectorstore_path,  # ✅ NEW
            "vectorstore_id": data["vectorstore_id"],
            "vectorstore_path": knowledge_vectorstore_path,
            "topic": data["topic"],
            "outline": data.get("outline", []),
            "knowledge_text": knowledge_text,  # Pre-loaded knowledge
            "knowledge_summary": report,

            "created_at": datetime.utcnow().isoformat(),
            "status": "active",  # active, completed
            "completed_count": 0,
            "total_count": len(data["candidates"])
        }

        # Save to MongoDB
        sessions_collection = db["interview_sessions_master"]
        result = sessions_collection.insert_one(session_doc)

        return jsonify({
            "success": True,
            "session_id": str(result.inserted_id),
            "message": "Buổi phỏng vấn đã được tạo thành công!"
        })

    except Exception as e:
        print(f"Error creating interview session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/interview_session/list", methods=["GET"])
def list_interview_sessions():
    """Lấy danh sách tất cả buổi phỏng vấn"""
    try:
        sessions_collection = db["interview_sessions_master"]
        sessions = list(sessions_collection.find().sort("created_at", -1))

        # Convert ObjectId to string
        for session in sessions:
            session["_id"] = str(session["_id"])
            # Don't send knowledge_text in list (too large)
            session.pop("knowledge_text", None)

        return jsonify({
            "success": True,
            "sessions": sessions
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/interview_session/detail/<session_id>")
def interview_session_detail(session_id):
    """Trang chi tiết buổi phỏng vấn"""

    return render_template("interview_session_detail.html", session_id=session_id)


@app.route("/interview_session/get/<session_id>", methods=["GET"])
def get_interview_session(session_id):
    """Lấy thông tin chi tiết buổi phỏng vấn"""
    try:
        sessions_collection = db["interview_sessions_master"]
        session = sessions_collection.find_one({"_id": ObjectId(session_id)})

        if not session:
            return jsonify({"success": False, "error": "Session not found"}), 404


        session["_id"] = str(session["_id"])
        # Don't send full knowledge_text unless requested
        # 2️⃣ Load CV vectorstore
        cv_path = session.get("cv_vectorstore_path")
        if cv_path:
            cv_db = FAISS.load_local(
                cv_path,
                global_interviewer.embeddings,
                allow_dangerous_deserialization=True
            )
            global_interviewer.cv_db = cv_db  # Cập nhật vectorstore thí sinh

        include_knowledge = request.args.get('include_knowledge', 'false') == 'true'
        if not include_knowledge:
            session.pop("knowledge_text", None)

        return jsonify({
            "success": True,
            "session": session
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/interview_session/start_candidate", methods=["POST"])
def start_candidate_interview():
    """
    Bắt đầu phỏng vấn cho một thí sinh trong buổi phỏng vấn
    Dùng knowledge_text đã preload trong session MongoDB.
    """
    global global_interviewer

    try:
        data = request.json
        session_id = data["session_id"]
        candidate_name = data["candidate_name"]
        candidate_class = data["candidate_class"]

        full_candidate_name = f"{candidate_name},{candidate_class}"

        sessions_collection = db["interview_sessions_master"]
        session = sessions_collection.find_one({"_id": ObjectId(session_id)})
        if not session:
            return jsonify({"error": "Session not found"}), 404

        config = InterviewConfig(**session["config"])
        global_interviewer.config = config

        # # Nếu vectorstore khác, reload knowledge builder
        # if session["vectorstore_path"] != getattr(global_interviewer.knowledge_builder.knowledge_db, "index_to_docstore_id", None):
        #     from langchain_community.vectorstores import FAISS
        #     embeddings = global_interviewer.embeddings
        #     knowledge_db = FAISS.load_local(
        #         session["vectorstore_path"],
        #         embeddings,
        #         allow_dangerous_deserialization=True
        #     )
        #     global_interviewer.knowledge_builder = KnowledgeBuilder(knowledge_db)
        outline_summary = session.get("knowledge_summary", "")
        # Gọi start_interview()
        result = global_interviewer.start_interview(
            candidate_name=full_candidate_name,
            topic=session["topic"],
            outline=session.get("outline", []),
            knowledge_text=session.get("knowledge_text", ""),
            outline_summary=outline_summary
        )

        # TTS
        question = result["question"]
        lang = detect_language(question)
        audio_id = create_audio_from_text(question, lang)

        result.update({
            "success": True,
            "audio_id": audio_id,
            "audio_url": f"/audio/{audio_id}" if audio_id else None,
            "phase": result.get("phase", "warmup")  # 👈 Thêm dòng này
        })

        return jsonify(result)

    except Exception as e:
        print(f"Error starting candidate interview: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500



@app.route("/interview_session/update_candidate_status", methods=["POST"])
def update_candidate_status():
    """Cập nhật trạng thái thí sinh sau khi hoàn thành phỏng vấn"""
    try:
        data = request.json
        session_id = data["session_id"]
        candidate_name = data["candidate_name"]
        status = data["status"]  # 'completed', 'in_progress', etc.

        sessions_collection = db["interview_sessions_master"]

        # Update candidate status in array
        sessions_collection.update_one(
            {
                "_id": ObjectId(session_id),
                "candidates.name": candidate_name
            },
            {
                "$set": {
                    "candidates.$.status": status,
                    "candidates.$.completed_at": datetime.utcnow().isoformat()
                }
            }
        )

        # Update completed count if status is 'completed'
        if status == 'completed':
            session = sessions_collection.find_one({"_id": ObjectId(session_id)})
            completed_count = sum(1 for c in session["candidates"] if c.get("status") == "completed")

            sessions_collection.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$set": {
                        "completed_count": completed_count,
                        "status": "completed" if completed_count == session["total_count"] else "active"
                    }
                }
            )

        return jsonify({"success": True})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/interview_session/delete/<session_id>", methods=["DELETE"])
def delete_interview_session(session_id):
    """Xóa buổi phỏng vấn"""
    try:
        sessions_collection = db["interview_sessions_master"]
        result = sessions_collection.delete_one({"_id": ObjectId(session_id)})

        if result.deleted_count > 0:
            return jsonify({"success": True})
        else:
            return jsonify({"success": False, "error": "Session not found"}), 404

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/interview_session/export/<session_id>", methods=["GET"])
def export_session_results(session_id):
    """Export kết quả buổi phỏng vấn ra CSV"""
    try:
        sessions_collection = db["interview_sessions_master"]
        session = sessions_collection.find_one({"_id": ObjectId(session_id)})

        if not session:
            return jsonify({"error": "Session not found"}), 404

        # Get all results for this session's candidates
        results_collection = db["interview_results"]
        results = []

        for candidate in session["candidates"]:
            candidate_full_name = f"{candidate['name']},{candidate['class']}"
            result = results_collection.find_one(
                {"candidate_info.name": candidate_full_name},
                sort=[("interview_stats.timestamp", -1)]
            )
            if result:
                results.append(result)

        # Generate CSV
        import csv
        from io import StringIO

        output = StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            'Tên', 'Lớp', 'Điểm cuối cùng', 'Số câu hỏi',
            'Trình độ', 'Thời gian', 'Trạng thái'
        ])

        # Data
        for result in results:
            name_parts = result["candidate_info"]["name"].split(",")
            writer.writerow([
                name_parts[0],
                name_parts[1] if len(name_parts) > 1 else "",
                result["interview_stats"].get("final_score", 0),
                result["interview_stats"].get("total_questions", 0),
                result["candidate_info"].get("classified_level", ""),
                result["interview_stats"].get("timestamp", ""),
                "Hoàn thành" if result else "Chưa hoàn thành"
            ])

        output.seek(0)

        return Response(
            output.getvalue(),
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment;filename=interview_results_{session_id}.csv"
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    # Khởi động background cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_scheduler, daemon=True)
    cleanup_thread.start()

    print("🚀 Server đang khởi động...")
    print("🔊 Text-to-Speech đã được kích hoạt")
    print("📁 Audio files được lưu tại:", AUDIO_FOLDER)

    app.run(debug=False, threaded=True)
    #Xóa file tạm khi kết thúc
    cleanup_temp_files()

