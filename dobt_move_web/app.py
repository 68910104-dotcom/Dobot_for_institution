import threading
import time
import serial.tools.list_ports
from flask import Flask, render_template, request, jsonify

# ==========================================
# 🔧 ตั้งค่า PORT ตรงนี้ (แก้เป็น COM ที่คุณเจอ)
# ==========================================
MANUAL_PORT = 'COM5'  # <--- เปลี่ยนเป็น COM3, COM4, หรือ COM5 ตามเครื่องคุณ
# ==========================================

# --- 1. Hardware Library ---
try:
    from pydobot import Dobot
except ImportError:
    print("❌ Critical Error: 'pydobot' library is missing. Please run: pip install pydobot")
    Dobot = None

app = Flask(__name__)

# --- 2. Controller Class ---
class DobotController:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DobotController, cls).__new__(cls)
            cls._instance.device = None
            cls._instance.connect()
        return cls._instance

    def connect(self):
        """เชื่อมต่อ Dobot"""
        if not Dobot: return

        # 1. ลองใช้ Port ที่ระบุมาเองก่อน (Manual)
        port = MANUAL_PORT
        
        # 2. ถ้าไม่ได้ระบุ ให้ลองหาเอง (Auto - ซึ่งอาจจะพลาดได้)
        if not port:
            port = self._find_port_windows()

        print(f"🔌 Attempting to connect to: {port} ...")

        if port:
            try:
                self.device = Dobot(port=port)
                self.device.speed(100, 100)
                print(f"✅ CONNECTED SUCCESS: Dobot found at {port}")
            except Exception as e:
                print(f"❌ CONNECTION FAILED at {port}: {e}")
                print("   -> ลองตรวจสอบสาย USB หรือปิดโปรแกรม Dobot Studio ดูครับ")
                self.device = None
        else:
            print("⚠️ DEVICE NOT FOUND: Please check USB connection.")
            self.device = None

    def _find_port_windows(self):
        ports = serial.tools.list_ports.comports()
        for port in ports:
            # พยายามหา Silicon Labs ก่อน
            if "Silicon Labs" in port.description or "CP210x" in port.description:
                return port.device
        # ถ้าไม่เจอให้เอา COM ตัวแรกที่เจอ
        for port in ports:
            if "COM" in port.device:
                return port.device
        return None

    def _check_connection(self):
        if self.device is None:
            # ถ้าหลุด ให้ลองต่อใหม่โดยใช้ logic เดิม
            self.connect()
        return self.device is not None

    def move_relative(self, dx=0, dy=0, dz=0, dr=0):
        with self._lock:
            if not self._check_connection(): return False, "Device not connected", (0,0,0,0)
            try:
                x, y, z, r, _, _, _, _ = self.device.pose()
                tx, ty, tz, tr = x+dx, y+dy, z+dz, r+dr
                self.device.move_to(tx, ty, tz, tr, wait=True)
                return True, "Move Success", (tx, ty, tz, tr)
            except Exception as e:
                return False, f"Error: {str(e)}", (0,0,0,0)

    def move_absolute(self, x, y, z, r):
        with self._lock:
            if not self._check_connection(): return False, "Device not connected", (0,0,0,0)
            try:
                self.device.move_to(x, y, z, r, wait=True)
                return True, "Move Success", (x, y, z, r)
            except Exception as e:
                return False, f"Error: {str(e)}", (0,0,0,0)

    def set_suction(self, enable):
        with self._lock:
            if not self._check_connection(): return False, "Device not connected", None
            try:
                self.device.suck(enable)
                return True, f"Suction {'ON' if enable else 'OFF'}", None
            except Exception as e:
                return False, f"Error: {str(e)}", None

    def get_pose(self):
        with self._lock:
            if not self._check_connection(): return (0,0,0,0,0,0,0,0)
            try:
                return self.device.pose()
            except:
                return (0,0,0,0,0,0,0,0)

# --- 3. Flask Server ---
bot = DobotController()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/command', methods=['POST'])
def command():
    data = request.json
    action = data.get('action')
    step = 20 
    
    success, msg = False, "Unknown Command"
    
    if action == 'up': success, msg, _ = bot.move_relative(dz=step)
    elif action == 'down': success, msg, _ = bot.move_relative(dz=-step)
    elif action == 'left': success, msg, _ = bot.move_relative(dy=-step)
    elif action == 'right': success, msg, _ = bot.move_relative(dy=step)
    elif action == 'front': success, msg, _ = bot.move_relative(dx=step)
    elif action == 'back': success, msg, _ = bot.move_relative(dx=-step)
    elif action == 'reset': success, msg, _ = bot.move_absolute(250, 0, 0, 0)
    elif action == 'suck_on': success, msg, _ = bot.set_suction(True)
    elif action == 'suck_off': success, msg, _ = bot.set_suction(False)
    elif action == 'move_to_coordinates':
        v = data.get('value', {})
        success, msg, _ = bot.move_absolute(
            float(v.get('x', 250)), float(v.get('y', 0)), 
            float(v.get('z', 0)), float(v.get('r', 0))
        )

    pose = bot.get_pose()
    return jsonify({ 
        "status": "success" if success else "error", 
        "message": msg, 
        "x": pose[0], "y": pose[1], "z": pose[2], "r": pose[3] 
    })

@app.route('/pose', methods=['GET'])
def get_pose_route():
    pose = bot.get_pose()
    return jsonify({ "x": pose[0], "y": pose[1], "z": pose[2], "r": pose[3] })

if __name__ == '__main__':
    # เพิ่ม use_reloader=False เพื่อป้องกันไม่ให้แย่ง Port กันเอง
    app.run(debug=True, port=5001, threaded=True, use_reloader=False)