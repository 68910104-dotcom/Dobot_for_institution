import serial.tools.list_ports
import pydobot
import time

# ----------------- Find Dobot Port -----------------
def find_dobot_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        # ตรวจจับชื่อพอร์ต Dobot Magician
        if "ttyACM" in p.device or "USB" in p.description:
            return p.device
    return None

port = find_dobot_port()
if not port:
    raise Exception("Dobot not found. Please check the connection.")

print("Found Dobot on:", port)

# ----------------- Connect to Dobot -----------------
device = pydobot.Dobot(port=port)

# ----------------- Get Pose Loop -----------------
try:
    while True:
        pose = device.pose()     # (x, y, z, r, j1, j2, j3, j4)
        print(f"X: {pose[0]:.2f}  Y: {pose[1]:.2f}  Z: {pose[2]:.2f}  R: {pose[3]:.2f}")
        time.sleep(0.2)

except KeyboardInterrupt:
    print("Exit.")

# ----------------- Disconnect -----------------
device.close()
