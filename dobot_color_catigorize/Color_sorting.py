import time
import cv2
import numpy as np
import serial.tools.list_ports
from pydobot import Dobot
# ไม่ต้องใช้ threading แล้ว เพราะเราจัดการกล้องแบบเปิด-ปิดชั่วคราว

# ----------------- Find Dobot Port -----------------
def find_dobot_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print("Found port:", p.device, "-", p.description)
        # ตรวจสอบพอร์ต Dobot
        if "ttyACM" in p.device or "ttyUSB" in p.device:
            return p.device
    return None

port = find_dobot_port()
if not port:
    raise Exception("Dobot not found. เช็คการเชื่อมต่อ USB หรือ COMXX บน Windows")

dobot = Dobot(port=port)
print("Connected to Dobot on:", port)

# ------------------------------------------------------
# 2) ตั้งค่าพิกัดของจุดต่างๆ
# ------------------------------------------------------
# จุดเริ่มต้น/พัก
start_point = {"x": 4, "y": 216, "z": 44}

# 3 ตำแหน่งดูดวัตถุ (ต้องปรับ Z ให้เป็นความสูงที่ดูดได้จริง)
pickup_points = [
    {"x": 63.22, "y": 268.80, "z": -56}, # จุดที่ 1
    {"x": 18.73, "y": 268.01, "z": -56}, # จุดที่ 2
    {"x": -25.45, "y": 271.92, "z": -56} # จุดที่ 3
]

# ตำแหน่งหน้ากล้องเพื่อเช็คสี
camera_point = {"x": 226.80, "y": 172.43, "z": -13.09}

# ตำแหน่งวางวัตถุตามสี (z=1 คือความสูงสำหรับวาง)
drop_points = {
    "Red":   {"x": -112.02, "y": 196.17, "z": 1},
    "Green": {"x": -7.62,   "y": 193.86,  "z": 1},
    "Blue":  {"x": 84.56,   "y": 185.00,  "z": 1}
}

# ------------------------------------------------------
# 3) ฟังก์ชันการตรวจจับสี (เปิด-อ่าน-ปิดกล้อง)
# ------------------------------------------------------
def detect_color_from_frame():
    # ใช้ Global cap เพื่อจัดการทรัพยากร
    cap = None 
    
    # พารามิเตอร์ตรวจจับ
    ROI_SIZE = 10 # รัศมี 10 พิกเซล (พื้นที่ 20x20)

    # 1. เปิดกล้องชั่วคราว
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[ERROR] Cannot open camera for color detection.")
            return "Unknown"
    
    # 2. อ่านภาพและปิดกล้อง
    ret, frame = cap.read()
    cap.release() 
    
    # 3. วิเคราะห์สี
    if not ret:
        print("[WARN] Failed to read frame for color detection.")
        return "Unknown"

    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    
    # กำหนด ROI
    y1 = max(0, cy - ROI_SIZE)
    y2 = min(h, cy + ROI_SIZE)
    x1 = max(0, cx - ROI_SIZE)
    x2 = min(w, cx + ROI_SIZE)
    
    roi_frame = frame[y1:y2, x1:x2]
    
    if roi_frame.size == 0:
        return "Unknown"

    hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    mean_hsv = np.mean(hsv_roi, axis=(0, 1))
    h_val = mean_hsv[0] 

    # 4. เกณฑ์การตรวจจับสี (Hue-based)
    if (h_val < 10) or (h_val > 160):
        return "Red"
    elif 35 <= h_val <= 85:
        return "Green"
    elif 90 <= h_val <= 130:
        return "Blue"
    else:
        return "Unknown"

# ------------------------------------------------------
# 4) ฟังก์ชันหลัก: pick → check → drop
# ------------------------------------------------------
def pick_and_sort(dobot, p_point):
    print("\n👉 3. ไปตำแหน่ง pickup:", p_point)

    # A. ไปเหนือจุดดูด (ปลอดภัย)
    dobot.move_to(p_point["x"], p_point["y"], p_point["z"] + 40, r=0, wait=True)
    # B. ลงไปที่จุดดูด
    dobot.move_to(p_point["x"], p_point["y"], p_point["z"],      r=0, wait=True)
    time.sleep(0.5)

    dobot.suck(True) # ดูดวัตถุ
    time.sleep(1) 

    # C. ยกขึ้น
    dobot.move_to(p_point["x"], p_point["y"], p_point["z"] + 55, r=0, wait=True)

    # 4. ไปตำแหน่งกล้องเพื่อเช็คสี
    print("👉 4. นำวัตถุไปหน้ากล้องเพื่อเช็คสี...")
    dobot.move_to(camera_point["x"], camera_point["y"], camera_point["z"], r=0, wait=True)

    time.sleep(1.0) # รอให้วัตถุหยุดนิ่ง
    
    color = detect_color_from_frame()
    print("   สีที่ตรวจพบ:", color)

    # 5. กลับไป start position ชั่วคราว (ตามที่คุณระบุ)
    print("👉 5. กลับไปจุดเริ่มต้นชั่วคราว...")
    dobot.move_to(start_point["x"], start_point["y"], start_point["z"], r=0, wait=True)


    # 6. ไปวางสีไว้เเต่ละกล่อง ของสีนั้นๆ 
    if color not in drop_points:
        print("   สีไม่รู้จัก → ปล่อยไว้ที่จุดเริ่มต้น")
        dobot.suck(False)
        return

    drop = drop_points[color]
    print(f"👉 6. นำไปวางสี {color} ที่กล่อง")

    # A. ไปเหนือจุด Drop
    dobot.move_to(drop["x"], drop["y"], drop["z"] + 40, r=0, wait=True)
    # B. ลงไปที่จุด Drop
    dobot.move_to(drop["x"], drop["y"], drop["z"],      r=0, wait=True)

    # 7. ปล่อย
    dobot.suck(False) 
    time.sleep(0.5)

    # C. ยกขึ้นจากจุด Drop
    dobot.move_to(drop["x"], drop["y"], drop["z"] + 40, r=0, wait=True)

# ------------------------------------------------------
# 5) Main Loop: ทำงานตามขั้นตอน 1, 2, 8, 9
# ------------------------------------------------------

# 1. เคลื่อนที่ไป start position
print("👉 1. เคลื่อนที่ไป Start Position...")
dobot.move_to(start_point["x"], start_point["y"], start_point["z"], r=0, wait=True)

try:
    while True:
        print("\n--- Control Panel ---")
        print("Enter 1 to start sorting cycle / Enter q to quit program")
        user_input = input(" : ")
        
        # 2. รอคำสั่ง 1 คือเริ่ม
        if user_input.lower() == 'q':
            print("\nShutting down by user input.")
            break

        if user_input == '1':
            print("--- Starting Sorting Cycle (8. ทำสีที่เหลือให้ครบทั้ง 3 จุด) ---")
            
            # วนลูปทำงานครบทั้ง 3 ตำแหน่งดูด
            for i, p in enumerate(pickup_points):
                print(f"\n--- Process Item {i+1} of {len(pickup_points)} ---")
                pick_and_sort(dobot, p)

            print("--- Sorting Cycle Finished ---")
            
            # 9. และกลับมา start position
            print("👉 9. กลับไป Start Position")
            dobot.move_to(start_point["x"], start_point["y"], start_point["z"], r=0, wait=True)
        else:
            # ถ้ากดอย่างอื่นไม่ต้องทำอะไร
            time.sleep(0.1) 

except Exception as e:
    print(f"\n[ERROR] An error occurred: {e}")

finally:
    # ------------------------------------------------------
    # ปิดระบบอย่างเรียบร้อย
    # ------------------------------------------------------
    print("\n[SYSTEM] Closing Dobot and Camera.")
    dobot.close()
    cv2.destroyAllWindows()
