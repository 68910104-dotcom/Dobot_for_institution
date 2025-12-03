from faster_whisper import WhisperModel
import os

def test_ai():
    print("--- เริ่มการทดสอบ AI ---")
    
    # 1. เช็คไฟล์เสียง
    filename = "record.wav"
    if not os.path.exists(filename):
        print(f"❌ ไม่เจอไฟล์ {filename} กรุณาอัดเสียงมาก่อน")
        return

    print(f"✅ เจอไฟล์ {filename}")

    # 2. โหลดโมเดล (บังคับใช้ CPU และ int8 เพื่อลดโอกาส Error)
    print("⏳ กำลังโหลดโมเดล (Force CPU)...")
    try:
        # ลองเปลี่ยนเป็น model size 'tiny' ก่อนเพื่อให้โหลดง่ายที่สุด
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ โหลดโมเดลสำเร็จ!")
    except Exception as e:
        print(f"❌ พังตอนโหลดโมเดล: {e}")
        return

    # 3. ลองถอดความ
    print("⏳ กำลังแปลงเสียงเป็นข้อความ...")
    try:
        segments, info = model.transcribe(filename, beam_size=5)
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        count = 0
        for segment in segments:
            print(f"💬 ผลลัพธ์: {segment.text}")
            count += 1
            
        if count == 0:
            print("⚠️ โมเดลทำงานได้ แต่จับใจความไม่ได้ (ลองพูดดังขึ้น)")
            
    except Exception as e:
        print(f"❌ พังตอนถอดความ: {e}")

if __name__ == "__main__":
    test_ai()