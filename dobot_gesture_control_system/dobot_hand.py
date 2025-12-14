import cv2
import mediapipe as mp
import time
from pydobot import Dobot
import serial.tools.list_ports

# --------------------- Dobot Setup ---------------------
def find_dobot_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description or "wchusbserial" in p.device.lower():
            return p.device
    return None

dobot_port = find_dobot_port()
if not dobot_port:
    raise Exception("ไม่พบ Dobot! กรุณาตรวจสอบการเชื่อมต่อ USB")

device = Dobot(port=dobot_port, verbose=False)
time.sleep(0.5)
device.move_to(x=220, y=0, z=50, r=0, wait=True)
device.suck(False)

# --------------------- MediaPipe Setup ---------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# --------------------- Finger Detection ---------------------
def finger_states(landmarks):
    thumb_up = landmarks[4].x < landmarks[3].x
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y
    return {
        'thumb': thumb_up,
        'index': index_up,
        'middle': middle_up,
        'ring': ring_up,
        'pinky': pinky_up
    }

# --------------------- Variables ---------------------
last_state = None
vacuum_on = False
default_z = 50   # ค่า Z คงที่

# --------------------- Main Loop ---------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        h, w, _ = frame.shape

        # Default position
        dobot_x, dobot_y, dobot_z = 50, 0, default_z

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                states = finger_states(landmarks)
                thumb = states['thumb']
                index = states['index']
                middle = states['middle']
                ring = states['ring']
                pinky = states['pinky']

                # ✌️ Movement mode (3 นิ้วกลาง+หัวแม่มือ)
                if thumb and index and middle and not ring and not pinky:
                    index_tip = landmarks[8]

                    # Mapping X (เข้า–ออก) จากแกน Y ของมือ
                    y_px = int(index_tip.y * h)
                    dobot_x = 100 + (h/2 - y_px) * 0.5
                    dobot_x = max(120, min(230, dobot_x))

                    # Mapping Y (ซ้าย–ขวา) จากแกน X ของมือ
                    x_px = int(index_tip.x * w)
                    dobot_y = (x_px - w/2) * 0.5
                    dobot_y = max(-150, min(150, dobot_y))

                    # Z คงที่
                    dobot_z = default_z

                    try:
                        device.move_to(x=dobot_x, y=dobot_y, z=dobot_z, r=0, wait=False)
                    except Exception as e:
                        print(f"Move Error: {e}")

                    if vacuum_on:
                        try:
                            device.suck(False)
                        except Exception as e:
                            print(f"Suck Off Error: {e}")
                        vacuum_on = False

                    last_state = 'move'

                # 🖐️ Hold (หยุดนิ่ง)
                elif all([thumb, index, middle, ring, pinky]):
                    if vacuum_on:
                        try:
                            device.suck(False)
                        except Exception as e:
                            print(f"Suck Off Error: {e}")
                        vacuum_on = False
                    last_state = 'hold'

                # ✊ Grab (ดูด → ยกขึ้น → เคลื่อนไป → ปล่อย)
                elif not any([thumb, index, middle, ring, pinky]):
                    if last_state != 'suck':
                        try:
                            device.suck(True)
                            vacuum_on = True

                            # ยกขึ้นเล็กน้อย
                            pose_values = device.pose()
                            x_current = pose_values[0]
                            y_current = pose_values[1]
                            r_current = pose_values[3] if len(pose_values) > 3 else 0
                            device.move_to(x=x_current, y=y_current, z=50, r=r_current, wait=True)

                            # เคลื่อนที่ไปปลายทาง (ตัวอย่าง)
                            device.move_to(x=x_current, y=y_current, z=-50, r=r_current, wait=True)
                            device.move_to(x=x_current, y=y_current, z=50, r=r_current, wait=True)
                            device.move_to(x=178.96, y=-148.89, z=50, r=0, wait=True)

                            # ปล่อยของ
                            device.suck(False)
                            vacuum_on = False

                        except Exception as e:
                            print(f"Suck On/Move Error: {e}")

                        last_state = 'suck'

        else:
            last_state = 'hold'

        # Display status
        cv2.putText(frame,
                    f"State: {last_state if last_state else 'None'}  Vacuum: {'ON' if vacuum_on else 'OFF'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame,
                    f"X:{int(dobot_x)} Y:{int(dobot_y)} Z:{int(dobot_z)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Dobot Control by Hand", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # กด Esc เพื่อออก
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    if 'device' in locals():
        device.close()
    print("👋 ปิดการเชื่อมต่อ Dobot และกล้องเรียบร้อยแล้ว")