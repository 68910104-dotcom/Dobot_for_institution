# --- ⭐️ 1. (เพิ่ม 2 บรรทัดนี้) บังคับใช้โหมด Agg (Non-GUI) ⭐️ ---
import matplotlib
matplotlib.use('Agg')
# --- จบส่วนที่เพิ่ม ---

import sys
import os

# --- ⭐️ 2. (คงไว้) โค้ดส่วนนี้แก้ปัญหา ModuleNotFoundError ⭐️ ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
# --- จบส่วนที่คงไว้ ---


# --- (โค้ด import เดิมของคุณ) ---
from resize512 import resize_image_to_512x512_and_convert
from draw_cartoon_df import process_folder_to_cartoon
from cropx4 import crop512
from combine import stitch_2x2_to_512
from remove_bg import rmbg_image

def dfcall(input_file_path):
    # (สคริปต์นี้ทำงานในโฟลเดอร์ /512_use/)
    
    print(f"[DFCall] 1. Removing BG from: {input_file_path}")
    rmbg_image(input_file_path)
    
    print("[DFCall] 2. Resizing...")
    input_resized = r"output_with_white_bg.png" 
    output_resized = "my_resized_512x512_image.jpg"
    resize_image_to_512x512_and_convert(input_resized, output_resized)
    
    print("[DFCall] 3. Cropping...")
    image_cropped = output_resized
    output_folder_cropped = "cropped_parts"
    os.makedirs(output_folder_cropped, exist_ok=True) 
    crop512(image_cropped, output_folder_cropped)
    
    print("[DFCall] 4. Cartoonizing...")
    input_folder_cartoon = output_folder_cropped
    output_folder_cartoon = "cartoon_output" 
    os.makedirs(output_folder_cartoon, exist_ok=True) 
    process_folder_to_cartoon(input_folder_cartoon, output_folder_cartoon)
    
    print("[DFCall] 5. Stitching...")
    input_folder_stitch = output_folder_cartoon
    stitch_2x2_to_512(input_folder_stitch)
    
    print("[DFCall] ✅ Done.")

# --- ⭐️ 3. (คงไว้) โค้ดส่วนนี้รับค่าจาก app.py ⭐️ ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <path_to_input_image>")
        sys.exit(1)
        
    image_path_from_server = sys.argv[1]
    
    print(f"--- Running DFCall for: {image_path_from_server} ---")
    dfcall(image_path_from_server)
    print(f"--- DFCall Finished ---")