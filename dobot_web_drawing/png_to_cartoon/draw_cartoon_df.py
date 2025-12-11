import matplotlib
matplotlib.use('Agg')

import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import sys

# --- Import Model ---
# ตรวจสอบ path ให้แน่ใจว่า models2/models.py อยู่ในตำแหน่งที่ Python หาเจอ
try:
    from models2.models import Generator
except ImportError:
    # กรณีหาไม่เจอ ลอง add path ปัจจุบัน
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from models2.models import Generator

def process_cartoon_gan(input_path, output_dir):
    """
    ฟังก์ชันหลัก: รับ input_path (เป็นไฟล์รูปเดียว หรือ โฟลเดอร์ก็ได้)
    และบันทึกผลลัพธ์ลงใน output_dir
    """
    
    # --- การตั้งค่าโมเดล (โหลดครั้งเดียว) ---
    #  แก้ไข path นี้ให้ตรงกับเครื่อง Server ของคุณ
    model_path = r"E:\P2LDGAN\p2ldgan_generator_200.pth" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cartoon] Using device: {device}")

    # 1. สร้าง Generator และโหลด Checkpoint
    try:
        generator = Generator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint)
        generator.eval()
        print("[Cartoon] Loaded checkpoint successfully.")
    except Exception as e:
        print(f"[Cartoon] Error loading model: {e}")
        return

    # 2. เตรียม Transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # โมเดลต้องการ 256x256
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 3. เตรียมโฟลเดอร์ Output
    os.makedirs(output_dir, exist_ok=True)

    # 4. ตรวจสอบว่า input เป็น ไฟล์เดียว หรือ โฟลเดอร์
    image_files = []
    if os.path.isfile(input_path):
        # กรณีรับมาเป็นไฟล์รูปเดียว (เช่น จากเว็บ)
        image_files.append(input_path)
    elif os.path.isdir(input_path):
        # กรณีรับมาเป็นโฟลเดอร์ (เช่น cropped_parts)
        for f in os.listdir(input_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(input_path, f))
    else:
        print(f"❌ [Cartoon] Input path not found: {input_path}")
        return

    print(f"[Cartoon] Found {len(image_files)} images to process.")

    # 5. เริ่มวนลูปประมวลผล
    for img_path in image_files:
        filename = os.path.basename(img_path)
        
        try:
            # โหลดรูปภาพ
            input_img = Image.open(img_path).convert("RGB")
            
            # แปลงเป็น Tensor
            input_tensor = transform(input_img).unsqueeze(0).to(device)

            # Generate (Inference)
            with torch.no_grad():
                output_tensor = generator(input_tensor)
                # Denormalize
                output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)

            # แปลงกลับเป็นรูปภาพ
            output_img = transforms.ToPILImage()(output_tensor.squeeze().cpu())
            
            # บันทึกไฟล์
            # ตั้งชื่อไฟล์ใหม่ เช่น cartoon_inputname.jpg
            save_name = f"cartoon_{filename}"
            output_filename = os.path.join(output_dir, save_name)
            
            output_img.save(output_filename)
            print(f"   Processed: {filename} -> {save_name}")
            
        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
            continue

    print(f"[Cartoon] All done. Saved to: {output_dir}")

# ---  ส่วนเชื่อมต่อกับ Web / Command Line  ---
if __name__ == '__main__':
    # การใช้งาน: python draw_cartoon_df.py <path_รูป_input> <path_โฟลเดอร์_output>
    
    if len(sys.argv) < 3:
        print("Usage: python draw_cartoon_df.py <input_path> <output_dir>")
        # ตัวอย่าง default ถ้าไม่ใส่ argument
        # input_arg = "cropped_parts"
        # output_arg = "cartoon_output"
        sys.exit(1)
    else:
        input_arg = sys.argv[1]  # รับค่า path รูป หรือ โฟลเดอร์ จาก argument ที่ 1
        output_arg = sys.argv[2] # รับค่า path output จาก argument ที่ 2

    print(f"--- Starting Cartoonizer ---")
    print(f"Input: {input_arg}")
    print(f"Output: {output_arg}")
    
    process_cartoon_gan(input_arg, output_arg)
