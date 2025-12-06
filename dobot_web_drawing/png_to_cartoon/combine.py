from PIL import Image
import os

def stitch_2x2_to_512(input_dir, output_filename="stitched_cartoon_512x512.jpg"):
    """
    ต่อรูปภาพขนาด 256x256 จำนวน 4 รูป (2x2) ให้เป็นรูปเดียวขนาด 512x512
    """
    
    TILE_SIZE = 256
    FINAL_SIZE = 512
    
    # 1. สร้างรูปภาพเปล่าขนาดสุดท้าย
    # 'RGB' คือโหมดสีมาตรฐาน, (FINAL_SIZE, FINAL_SIZE) คือขนาด (512, 512)
    final_img = Image.new('RGB', (FINAL_SIZE, FINAL_SIZE))

    # 2. กำหนดชื่อไฟล์รูปย่อยและตำแหน่งสำหรับวาง (Paste Position)
    # เราจะใช้โครงสร้างตาราง 2x2: (แถว, คอลัมน์) -> (ชื่อไฟล์, พิกัด x, พิกัด y)
    
    # พิกัด x และ y สำหรับ 4 ตำแหน่ง: (0, 0), (256, 0), (0, 256), (256, 256)
    tile_data = [
        # ตำแหน่งบนซ้าย (r=0, c=0)
        {"filename": "cartoon_part_1.jpg", "x": 0, "y": 0},
        # ตำแหน่งบนขวา (r=0, c=1)
        {"filename": "cartoon_part_2.jpg", "x": TILE_SIZE, "y": 0},
        # ตำแหน่งล่างซ้าย (r=1, c=0)
        {"filename": "cartoon_part_3.jpg", "x": 0, "y": TILE_SIZE},
        # ตำแหน่งล่างขวา (r=1, c=1)
        {"filename": "cartoon_part_4.jpg", "x": TILE_SIZE, "y": TILE_SIZE},
    ]

    print(f"กำลังต่อรูปภาพขนาด {TILE_SIZE}x{TILE_SIZE} จำนวน 4 รูป...")

    # 3. วนลูปเพื่อโหลดและวางรูปภาพ
    for i, data in enumerate(tile_data):
        tile_path = os.path.join(input_dir, data["filename"])
        
        try:
            tile_img = Image.open(tile_path)
        except FileNotFoundError:
            print(f"❗ คำเตือน: ไม่พบไฟล์ {data['filename']} ใน '{input_dir}' ข้ามไป.")
            continue
            
        paste_position = (data["x"], data["y"])
        
        # วางรูปภาพย่อยลงบนรูปภาพหลักที่ตำแหน่งที่กำหนด
        final_img.paste(tile_img, paste_position)
        
        print(f"  > วางส่วนที่ {i+1} ({data['filename']}) ที่ตำแหน่ง {paste_position}")

    # 4. บันทึกรูปภาพที่ต่อสมบูรณ์แล้ว
    final_img.save(output_filename)

    print(f"\n✅ การต่อรูปภาพเสร็จสมบูรณ์! บันทึกเป็น '{output_filename}'")
    
# --- ตัวอย่างการเรียกใช้ฟังก์ชัน ---
if __name__ == '__main__':
    # กำหนดโฟลเดอร์ที่มีรูปภาพการ์ตูน 256x256 อยู่
    input_folder = "cartoon_output" 
    
    # กำหนดชื่อไฟล์ผลลัพธ์
    output_name = "reconstructed_cartoon_final.jpg"
    
    stitch_2x2_to_512(input_folder, output_name)