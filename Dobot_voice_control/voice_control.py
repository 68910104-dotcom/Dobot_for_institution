# voice_control.py (V3: Fuzzy Logic + Smart Matching)
import time
import wave
import numpy as np
import sounddevice as sd
from pydobot import Dobot
from difflib import get_close_matches # <--- พระเอกของเรา (ช่วยหาคำใกล้เคียง)

# เรียกใช้ฟังก์ชันถอดเสียง
from NLP import transcribe_wav 

# --- การตั้งค่า ---
PORT = "COM3"
STEP = 20
MEMORY_POINTS = []

# ==========================================
# 🗺️ แผนที่คำสั่ง (รวมคำผิดที่พบบ่อย)
# ==========================================
CMD_MAP = {
    # --- ทิศทาง (Move) ---
    "ซ้าย": "left", "left": "left", "lift": "left", "leaf": "left", "deaf": "left",
    "ขวา": "right", "right": "right", "write": "right", "light": "right", "white": "right", "ride": "right", "like": "right",
    "ขึ้น": "up", "up": "up", "app": "up", "pub": "up", "hub": "up",
    "ลง": "down", "down": "down", "dawn": "down", "don": "down",
    "หน้า": "front", "front": "front", "forward": "front", "for": "front", "four": "front",
    "หลัง": "back", "back": "back", "bag": "back", "black": "back", "beg": "back",
    
    # --- ระบบดูด (Suction) ---
    "ดูด": "suck", "suck": "suck", "sock": "suck", "sack": "suck", "pick": "suck", "grab": "suck",
    "ปล่อย": "release", "วาง": "release", "release": "release", "realize": "release", "drop": "release",

    # --- ความจำ (Memory) ---
    "บันทึก": "save", "จำ": "save", "save": "save", "safe": "save", "keep": "save",
    "เล่น": "play", "วน": "play", "play": "play", "pay": "play", "plate": "play", "replay": "play",
    "ล้าง": "clear", "ลบ": "clear", "clear": "clear", "clean": "clear"
}

# ==============================
# 1) ฟังก์ชันอัดเสียง
# ==============================
def record_until_enter(filename: str = "record.wav", samplerate: int = 16000, channels: int = 1) -> str:
    print("\n" + "-"*30)
    input("🎤 กด Enter เพื่ออัดเสียง... (แล้วพูดคำสั่ง)")
    print("⏺️  กำลังรับฟัง... (พูดจบแล้วกด Enter)")
    frames = []

    def callback(indata, frames_count, time_info, status):
        if status: print(status)
        frames.append(indata.copy())

    stream = sd.InputStream(samplerate=samplerate, channels=channels, callback=callback)
    stream.start()
    input()
    stream.stop()
    stream.close()

    audio_data = np.concatenate(frames, axis=0) if frames else np.zeros((1, channels), dtype=np.float32)
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    return filename

# ==============================
# 2) ฟังก์ชันสมองกล (Smart Command)
# ==============================
def normalize_command(text: str):
    """
    ค้นหาคำสั่งแบบ Fuzzy (ใกล้เคียงก็รับ)
    """
    if not text: return None
    
    # 1. แปลงเป็นตัวพิมพ์เล็ก และแยกเป็นคำๆ
    words = text.lower().split()
    all_keys = list(CMD_MAP.keys())

    print(f"🔍 วิเคราะห์คำ: {words}")

    for word in words:
        # A. เช็คแบบตรงตัว (Exact Match)
        if word in CMD_MAP:
            return CMD_MAP[word]
        
        # B. เช็คแบบใกล้เคียง (Fuzzy Match) 
        # cutoff=0.7 แปลว่า เหมือนกัน 70% ขึ้นไปถึงจะเอา (เช่น 'writting' -> 'right' อาจจะไม่ถึง แต่ 'write' -> 'right' ถึง)
        matches = get_close_matches(word, all_keys, n=1, cutoff=0.7)
        if matches:
            best_match = matches[0]
            print(f"✨ เจอคำใกล้เคียง: '{word}' ≈ '{best_match}'")
            return CMD_MAP[best_match]

    return None

def get_xyzr(device):
    pose = device.pose()
    return pose[:4]

def move_relative(device, direction, step=STEP):
    vec = {
        "left":  (0, -1,  0, 0), "right": (0,  1,  0, 0),
        "front": (1,  0,  0, 0), "back":  (-1, 0,  0, 0),
        "up":    (0,  0,  1, 0), "down":  (0,  0, -1, 0),
    }.get(direction)

    if vec:
        dx, dy, dz, dr = vec
        x, y, z, r = get_xyzr(device)
        device.move_to(x + dx*step, y + dy*step, z + dz*step, r + dr*step)
        print(f"✅ ขยับ {direction}")

def connect_dobot(port=PORT):
    print(f"🚀 เชื่อมต่อ {port}...")
    try:
        device = Dobot(port=port, verbose=False)
        time.sleep(1)
        try:
            device.ser.reset_input_buffer()
            device.ser.reset_output_buffer()
        except: pass
        print("✅ Ready!")
        return device
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

# ==============================
# 3) Main Loop
# ==============================
def main():
    device = connect_dobot(PORT)
    if not device: return

    print("\n🎧 Voice V3 (Smart Fuzzy Mode)")
    print("รองรับคำเพี้ยน เช่น Write->Right, App->Up")

    try:
        while True:
            wav = record_until_enter()
            text = transcribe_wav(wav)
            
            if not text: 
                print("🤷‍♂️ เงียบกริบ...")
                continue

            print(f"🗣️  ได้ยิน: '{text}'")
            cmd = normalize_command(text)

            if not cmd:
                print("🤔 ไม่เข้าใจ (ลองพูดใหม่)")
                continue

            print(f"🤖 สั่งงาน: {cmd.upper()}")

            # --- ส่วนจัดการคำสั่ง (เหมือนเดิม) ---
            if cmd in ["left", "right", "up", "down", "front", "back"]:
                move_relative(device, cmd, STEP)

            elif cmd == "suck":
                device.suck(True)
                print("💨 ดูดจ๊วบ!")
            elif cmd == "release":
                device.suck(False)
                print("🍃 ปล่อยของ")

            elif cmd == "save":
                pos = get_xyzr(device)
                MEMORY_POINTS.append(pos)
                print(f"💾 จำจุดที่ {len(MEMORY_POINTS)}")
            
            elif cmd == "clear":
                MEMORY_POINTS.clear()
                print("🗑️ ล้างสมองแล้ว")

            elif cmd == "play":
                if not MEMORY_POINTS:
                    print("⚠️ ไม่มีข้อมูลให้เล่น")
                else:
                    print(f"▶️ Replay {len(MEMORY_POINTS)} จุด...")
                    for i, pos in enumerate(MEMORY_POINTS):
                        print(f"   -> จุด {i+1}")
                        device.move_to(pos[0], pos[1], pos[2], pos[3], wait=True)
                    print("✅ จบ")

    except KeyboardInterrupt:
        print("\n👋 บาย")
    finally:
        if device: device.close()

if __name__ == "__main__":
    main()