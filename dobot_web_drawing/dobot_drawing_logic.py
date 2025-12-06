# --- ⭐️ บังคับใช้โหมด Agg (Non-GUI) เพื่อป้องกันเซิร์ฟเวอร์แครช ⭐️ ---
import matplotlib
matplotlib.use('Agg')
# --------------------------------------------------

import cv2
import numpy as np
try:
    from pydobot import Dobot
except ImportError:
    Dobot = None

import time
import os
import matplotlib.pyplot as plt
import json 
import glob
import sys
import shutil
import math

# ================== CONFIG (สูตรเน้นความเร็ว) ==================
OUTPUT_DIR_BASE = 'static/processed' 
EXP_PREFIX = 'exp_' 
# ---------------------------------------------

IMAGE_MAX_SIZE = 1000

# ระดับปากกา (ปรับตามความเหมาะสมของเครื่องคุณ)
PEN_DOWN_Z = -45   # จุดจรดกระดาษ
PEN_UP_Z = -40     # ⭐️ ยกขึ้นแค่นิดเดียว (5mm) เพื่อความเร็ว

RETRY_ATTEMPTS = 3

# ความเร็ว (ตั้งไว้สูงๆ)
DOBOT_SPEED = 4000      
DOBOT_ACCELERATION = 3000

# ⭐️ ลดความละเอียดลงนิดนึง เพื่อให้วาดลื่นไหล (ไม่กระตุก)
EPSILON = 0.0020

# ⭐️ ตัดเส้นขยะเล็กๆ ทิ้งไปเลย
MIN_CONTOUR_AREA = 50

# ⭐️ ระยะห่างที่จะดูดเส้นเข้าหากัน (Pixel)
# เพิ่มเป็น 80 เพื่อให้เส้นที่ขาดช่วงไกลๆ เชื่อมกันได้ (ลดการยกปากกา)
MERGE_DISTANCE_THRESHOLD = 1 

# ⭐️ ความหนาแน่นของการถมดำ (3 = เร็วขึ้นมาก, 1 = ละเอียดช้า)
FILL_DENSITY = 3

# ขนาดพื้นที่ที่จะตัดสินว่าเป็น "ตา/จมูก"
EYE_AREA_MAX_THRESHOLD = 400 

# Preset Parameters: (Name, Blur, Block, C, Epsilon, MinArea)
TEST_PARAMS = [
    ("Smart Hybrid (Fast)", 3, 9, 4, 0.0020, 50),    # ⭐️ แนะนำอันนี้
    ("Detail Focus", 3, 7, 2, 0.0010, 10),      
    ("Smooth Lines", 5, 11, 5, 0.0015, 20),    
    ("Thick Lines", 5, 15, 5, 0.0015, 25),    
    ("Simple", 7, 21, 8, 0.0030, 50)
]

CALIBRATION_FILE = 'dobot_calibration.json'

PAPER_CORNERS_DEFAULT = np.float32([
    [1.69, 96.04],      # มุมบนซ้าย
    [134.10, 215.25],   # มุมบนขวา
    [264.16, 28.42],    # มุมล่างขวา
    [106.29, -51.89]    # มุมล่างซ้าย
])

# ----------------- ฟังก์ชันช่วยเหลือทั่วไป -----------------

def load_calibration():
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, 'r') as f:
                corners_list = json.load(f)
                if len(corners_list) == 4 and all(len(c) == 2 for c in corners_list):
                    print(f"✅ โหลดค่า Calibration จาก {CALIBRATION_FILE}")
                    return np.float32(corners_list)
        except Exception:
            pass
    return PAPER_CORNERS_DEFAULT

PAPER_CORNERS = load_calibration()

def find_dobot_port():
    import serial.tools.list_ports
    ports = serial.tools.list_ports.comports()
    print("\n🔍 DEBUG: รายชื่อ Port ที่เจอ (Auto):")
    for p in ports:
        print(f"   - Device: {p.device}, Desc: {p.description}")
        if not hasattr(p, 'description') or not hasattr(p, 'device'): continue
        is_dobot = "USB" in p.description.upper() or \
                   "SERIAL" in p.description.upper() or \
                   "CH340" in p.description.upper() or \
                   "CP210" in p.description.upper() or \
                   "USB" in p.device.upper()
        if is_dobot:
            print(f"✅ เลือกใช้ Port: {p.device}")
            return p.device
    print("❌ ไม่พบ Port ที่เข้าข่าย")
    return None

def safe_move(bot, x, y, z, r=0, wait=True):
    for i in range(RETRY_ATTEMPTS):
        try:
            bot.move_to(x, y, z, r, wait=wait)
            return True
        except Exception:
            time.sleep(0.1)
    return False

def get_next_experiment_dir():
    os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}[0-9]*'))
    max_num = 0
    for dir_path in existing_dirs:
        try:
            num_str = os.path.basename(dir_path).replace(EXP_PREFIX, '')
            max_num = max(max_num, int(num_str))
        except ValueError:
            continue
    next_num = max_num + 1
    new_exp_dir = os.path.join(OUTPUT_DIR_BASE, f'{EXP_PREFIX}{next_num}')
    
    os.makedirs(os.path.join(new_exp_dir, 'all_steps'), exist_ok=True)
    os.makedirs(os.path.join(new_exp_dir, 'current_run'), exist_ok=True)
    
    print(f"✅ สร้างโฟลเดอร์งานใหม่: {new_exp_dir}/")
    return new_exp_dir 

def create_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final, 
                          output_all_steps_path, output_current_run_path):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    
    if not is_final:
        filename_all = os.path.join(output_all_steps_path, f"step_{current_contour_index:04d}_drawing.jpg")
        cv2.imwrite(filename_all, preview)
        
    filename_current = os.path.join(output_current_run_path, f"current_progress_{'done' if is_final else 'drawing'}.jpg")
    cv2.imwrite(filename_current, preview)

def update_current_progress_image(base_img_bgr, filtered_contours, current_contour_index, is_final, output_filename):
    preview = base_img_bgr.copy()
    if current_contour_index > 1:
        cv2.drawContours(preview, filtered_contours[:current_contour_index-1], -1, (255, 0, 0), 1) 
    if not is_final and current_contour_index <= len(filtered_contours):
        cv2.drawContours(preview, [filtered_contours[current_contour_index-1]], -1, (0, 255, 0), 2)
    cv2.imwrite(output_filename, preview)

# --- ⭐️ Helper Functions ⭐️ ---

def skeletonize(img):
    """บีบเส้นหนาๆ ให้เหลือแกนกลางเพียง 1 พิกเซล"""
    img = img.copy() 
    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    while True:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0: break
    return skel

def simplify_path_rdp(path, epsilon=2.0):
    """ลดจุดยิบย่อยบนเส้นตรง ทำให้ลากยาวๆ ได้"""
    if len(path) < 3: return path
    return cv2.approxPolyDP(path, epsilon, False)

def generate_concentric_fill(binary_mask, step_size=FILL_DENSITY):
    """ถมดำพื้นที่ตา/จมูก แบบวนก้นหอย"""
    fill_contours = []
    temp_mask = binary_mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    loop_count = 0
    while True:
        if loop_count >= 100: break
        contours, _ = cv2.findContours(temp_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: break
        added_any = False
        for cnt in contours:
            if cv2.contourArea(cnt) > 2: 
                # Simplify ให้เส้นน้อยลง (วาดเร็วขึ้น)
                approx = cv2.approxPolyDP(cnt, 1.0, False)
                fill_contours.append(approx)
                added_any = True
        if not added_any: break
        for _ in range(step_size): temp_mask = cv2.erode(temp_mask, kernel)
        if cv2.countNonZero(temp_mask) == 0: break
        loop_count += 1
    return fill_contours

def sort_and_merge_contours(contours, threshold=MERGE_DISTANCE_THRESHOLD):
    """
    จัดลำดับและเชื่อมเส้น (Optimization)
    """
    if not contours: return []
    unvisited = [c for c in contours]
    ordered_paths = []
    
    # เริ่มจากเส้นที่ใหญ่ที่สุด (โครงหน้า)
    unvisited.sort(key=lambda x: cv2.arcLength(x, False), reverse=True)
    current_path = unvisited.pop(0)
    
    while True:
        current_end_point = current_path[-1][0]
        best_dist = float('inf')
        best_idx = -1
        should_reverse = False
        
        for i, p in enumerate(unvisited):
            start_p = p[0][0]
            end_p = p[-1][0]
            dist_start = np.linalg.norm(current_end_point - start_p)
            dist_end = np.linalg.norm(current_end_point - end_p)
            
            if dist_start < best_dist:
                best_dist = dist_start; best_idx = i; should_reverse = False
            if dist_end < best_dist:
                best_dist = dist_end; best_idx = i; should_reverse = True
        
        if best_idx != -1:
            next_path = unvisited[best_idx]
            if best_dist < threshold:
                # เชื่อมเส้น (Merge)
                if should_reverse: next_path = next_path[::-1]
                current_path = np.vstack((current_path, next_path))
                unvisited.pop(best_idx)
            else:
                # จบเส้นนี้ (Simplify แล้วเก็บ)
                current_path = simplify_path_rdp(current_path, epsilon=2.0)
                ordered_paths.append(current_path)
                current_path = unvisited.pop(best_idx)
                if should_reverse: current_path = current_path[::-1]
        else:
            current_path = simplify_path_rdp(current_path, epsilon=2.0)
            ordered_paths.append(current_path)
            if unvisited: current_path = unvisited.pop(0)
            else: break
    return ordered_paths

# --- ⭐️ LOGIC หลัก (สูตร Fast + High Quality) ⭐️ ---
# ปรับปรุงให้รับพารามิเตอร์ปรับแต่งได้
def process_and_draw_contours(img_gray, blur_ksize, thresh_blocksize, thresh_c, epsilon_factor, min_contour_area, merge_threshold=MERGE_DISTANCE_THRESHOLD):
    
    # 1. เร่ง Contrast (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_gray)

    # 2. Blur
    if blur_ksize % 2 == 0: blur_ksize += 1
    img_blurred = cv2.GaussianBlur(img_enhanced, (blur_ksize, blur_ksize), 0)
    
    # 3. Adaptive Threshold
    if thresh_blocksize % 2 == 0: thresh_blocksize += 1
    if thresh_blocksize < 3: thresh_blocksize = 3
    thresh = cv2.adaptiveThreshold(
        img_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, thresh_blocksize, thresh_c
    )
    
    # 4. แยกเลเยอร์ (เตรียมข้อมูล)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thresh_filled = cv2.dilate(thresh, kernel_dilate, iterations=1)
    
    final_contours = []
    
    # --- Layer A: Body (Skeletonize -> เส้นเดียว) ---
    body_skeleton = skeletonize(thresh_filled)
    body_contours, _ = cv2.findContours(body_skeleton, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in body_contours:
        area = cv2.contourArea(cnt)
        length = cv2.arcLength(cnt, False)
        # กรองเส้นขยะที่เล็กมากๆ ทิ้งไป (ใช้ min_contour_area ที่รับมา)
        if length > 25 or area > min_contour_area:
            # ใช้ epsilon_factor ที่รับมา
            approx = cv2.approxPolyDP(cnt, epsilon_factor * length, False)
            final_contours.append(approx)

    # --- Layer B: Eyes/Details (Fill -> ถมดำ) ---
    _, mask_details = cv2.threshold(img_blurred, 90, 255, cv2.THRESH_BINARY_INV)
    mask_details = cv2.morphologyEx(mask_details, cv2.MORPH_OPEN, np.ones((2,2), np.uint8)) 
    
    detail_contours, _ = cv2.findContours(mask_details, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_eyes_fill = np.zeros_like(thresh)
    
    for cnt in detail_contours:
        area = cv2.contourArea(cnt)
        # กรองเฉพาะตา/จมูก
        if 15 < area < EYE_AREA_MAX_THRESHOLD:
            cv2.drawContours(mask_eyes_fill, [cnt], -1, 255, -1)
    
    # ถมดำแบบเร็ว (FILL_DENSITY = 3)
    eye_fill_lines = generate_concentric_fill(mask_eyes_fill, step_size=FILL_DENSITY)
    final_contours.extend(eye_fill_lines)

    # 5. Optimize (รวมเส้น + เรียงลำดับ + ทำให้เส้นตรง)
    # ส่ง merge_threshold ที่รับมาไปใช้
    optimized_contours = sort_and_merge_contours(final_contours, threshold=merge_threshold)
    
    preview_img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(preview_img_bgr, optimized_contours, -1, (0, 0, 255), 1)
    
    total_len_pixel = sum([cv2.arcLength(c, False) for c in optimized_contours])

    return preview_img_bgr, optimized_contours, total_len_pixel

def visualize_parameters(original_img_color, original_img_gray, test_params, output_dir):
    fig, axs = plt.subplots(3, 2, figsize=(8.27, 11.69)) 
    axs = axs.flatten()
    axs[0].imshow(cv2.cvtColor(original_img_color, cv2.COLOR_BGR2RGB))
    axs[0].set_title("1. Original Image (BGR)", fontsize=10, fontweight='bold')
    axs[0].axis("off")
    
    all_test_params = TEST_PARAMS
    for i, (name, blur, block, c, eps, min_area) in enumerate(all_test_params, start=1):
        if i >= len(axs): break
        processed_img_bgr, _, _ = process_and_draw_contours(
            original_img_gray.copy(), blur, block, c, eps, min_area
        )
        axs[i].imshow(cv2.cvtColor(processed_img_bgr, cv2.COLOR_BGR2RGB))
        axs[i].set_title(f"{i+1}. {name}", fontsize=8)
        axs[i].axis("off")
        
    for i in range(len(all_test_params) + 1, len(axs)): fig.delaxes(axs[i])
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.suptitle("Dobot Drawing Parameter Comparison", fontsize=16, fontweight='bold')
    output_filename = os.path.join(output_dir, "parameter_comparison.jpg")
    plt.savefig(output_filename, dpi=200) 
    plt.close(fig) 
    print(f"✅ บันทึกภาพเปรียบเทียบที่: {output_filename}")
    return output_filename 

def get_eta_display(start_time, current_length_drawn, total_length_to_draw):
    elapsed_time = time.time() - start_time
    if elapsed_time > 5 and current_length_drawn > 10 and current_length_drawn < total_length_to_draw: 
        try:
            avg = current_length_drawn / elapsed_time 
            rem = total_length_to_draw - current_length_drawn
            return f"ETA: {rem/avg/60:.1f} min"
        except: pass
    elif current_length_drawn >= total_length_to_draw: return "ETA: Done"
    return "ETA: Calculating..."

print("✅ dobot_drawing_logic.py loaded.")