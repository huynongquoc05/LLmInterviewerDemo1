from gtts import gTTS
import os
text='''Trong Java, các kiểu dữ liệu cơ sở (primitive types) như `int`, `double` và các lớp Wrapper tương ứng (như `Integer`, `Double`) đều có thể lưu trữ giá trị. Vậy trong những tình huống ứng dụng thực tế nào, việc sử dụng các lớp Wrapper được ưu tiên hơn so với kiểu dữ liệu cơ sở, và tại sao?
'''
tts = gTTS(text, lang="vi")
tts.save("output1.mp3")
os.system("start output1.mp3")

# from gtts import gTTS
# from io import BytesIO
# from pydub import AudioSegment
# from pydub.playback import play
#
# text = "Xin chào, mình đang phát trực tiếp không lưu file."
# tts = gTTS(text, lang="vi")
# fp = BytesIO()
# tts.write_to_fp(fp)  # ghi thẳng dữ liệu vào bộ nhớ
# fp.seek(0)
#
# song = AudioSegment.from_file(fp, format="mp3")
# play(song)

