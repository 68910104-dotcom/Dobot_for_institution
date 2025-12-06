from PIL import Image

def crop512(input_filename,output_dir):
    
    try:
        img = Image.open(input_filename)
    except FileNotFoundError:
        print(f"ไม่พบไฟล์: {input_filename}")
        exit()

    # 2. กำหนดขนาดและพิกัดเริ่มต้น
    tile_size = 256  # ขนาดของรูปย่อยที่ต้องการ (256x256)
     # กำหนดโฟลเดอร์สำหรับบันทึก

    # 3. สร้างรายการพิกัดสำหรับ 4 ส่วน
    # (left, upper, right, lower)
    crop_boxes = [
        (0, 0, tile_size, tile_size),              # บนซ้าย
        (tile_size, 0, 512, tile_size),            # บนขวา
        (0, tile_size, tile_size, 512),            # ล่างซ้าย
        (tile_size, tile_size, 512, 512)           # ล่างขวา
    ]

    # 4. ทำการ Crop และบันทึก
    import os
    os.makedirs(output_dir, exist_ok=True) # ตรวจสอบและสร้างโฟลเดอร์

    for i, box in enumerate(crop_boxes):
        # ทำการ Crop
        cropped_img = img.crop(box)

        # กำหนดชื่อไฟล์สำหรับบันทึก
        output_filename = os.path.join(output_dir, f"part_{i+1}.jpg")
        
        # บันทึกรูปภาพ
        cropped_img.save(output_filename)
        print(f"บันทึกส่วนที่ {i+1} ({box}) -> {output_filename}")
        
    print("\n✅ การ Crop แบ่ง 4 ส่วนเสร็จสมบูรณ์!")