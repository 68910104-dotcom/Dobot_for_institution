# NLP.py (ฉบับ Hybrid: ฟังไทย + อังกฤษ)
import speech_recognition as sr

def transcribe_wav(filename="record.wav"):
    recognizer = sr.Recognizer()
    
    # อ่านไฟล์เสียง
    with sr.AudioFile(filename) as source:
        print(f"กำลังส่งเสียงไปถอดความที่ Google (Thai & English)...")
        audio_data = recognizer.record(source)
        
    result_text = ""
    
    # 1. ลองถอดความภาษาไทย (th-TH)
    try:
        text_th = recognizer.recognize_google(audio_data, language="th-TH")
        result_text += text_th + " " 
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        print("เชื่อมต่อ Google (Thai) ไม่ได้")

    # 2. ลองถอดความภาษาอังกฤษ (en-US)
    try:
        text_en = recognizer.recognize_google(audio_data, language="en-US")
        result_text += text_en
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        print("เชื่อมต่อ Google (English) ไม่ได้")

    # ส่งผลลัพธ์กลับไปทั้งคู่เลย (เช่น "ซ้าย Left")
    print(f"ผลลัพธ์รวม: {result_text}")
    return result_text