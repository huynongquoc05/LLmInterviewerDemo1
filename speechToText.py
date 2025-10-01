import time
def speech_to_text_by_whisper(model_name: str):

    import whisper
    statr_time = time.time()
    model = whisper.load_model(model_name).to('cuda')
    result = model.transcribe("output.mp3", language="vi", task="transcribe")
    print(result["text"])
    print(f'thời gian thực thi bằng model whisper {model_name}',time.time() - statr_time)


def speech_to_text_by_speech_recog():

    import speech_recognition as sr
    from pydub import AudioSegment
    start_time = time.time()
    # B1: Convert mp3 -> wav
    audio_mp3 = "output.mp3"
    audio_wav = "output.wav"
    AudioSegment.from_mp3(audio_mp3).export(audio_wav, format="wav")

    # B2: Khởi tạo recognizer
    recognizer = sr.Recognizer()

    # B3: Load file âm thanh
    with sr.AudioFile(audio_wav) as source:
        audio = recognizer.record(source)  # đọc toàn bộ file

    # B4: Nhận diện giọng nói bằng Google Speech Recognition (miễn phí, online)
    try:
        text = recognizer.recognize_google(audio, language="vi-VN")
        print("Kết quả nhận diện:", text)
    except sr.UnknownValueError:
        print("Google Speech Recognition không hiểu được audio")
    except sr.RequestError as e:
        print(f"Lỗi kết nối tới Google Speech Recognition: {e}")
    print('thời gian thực thi',time.time() - start_time)

# import speech_recognition as sr
#
# r = sr.Recognizer()
# with sr.Microphone() as source:
#     print("Nói gì đó...")
#     audio = r.listen(source)
#
# try:
#     text = r.recognize_google(audio, language="vi-VN")
#     print("Bạn vừa nói:", text)
# except Exception as e:
#     print("Lỗi:", e)
# # from pydub import AudioSegment

if __name__ == '__main__':
    speech_to_text_by_whisper('small')
    speech_to_text_by_whisper('medium')
    speech_to_text_by_speech_recog()
