from pydobot import Dobot
import time

port = 'COM3' 
device = Dobot(port=port, verbose=True)

try:
    (x, y, z, r, j1, j2, j3, j4) = device.pose()
    print(f"Current position: x:{x} y:{y} z:{z}")

    device.move_to(200, 0, 50, 0, wait=True)
    time.sleep(1)
    
    device.move_to(200, 50, 50, 0, wait=True)
    
    print("เคลื่อนที่เสร็จสิ้น")

finally:
    device.close()