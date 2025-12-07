import sys
import os

# --- ⭐️ แก้ไข: ให้มองหาไฟล์ในโฟลเดอร์ "แม่" (Parent Directory) ⭐️ ---
current_dir = os.path.dirname(os.path.abspath(__file__)) # ได้ path ของ /512_use
parent_dir = os.path.dirname(current_dir) # ได้ path ของ /dfcall (ถอยขึ้น 1 ชั้น)

# เพิ่ม path ของ /dfcall เข้าไปในระบบ เพื่อให้ import models2 ได้
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ----------------------------------------------------------------

import cv2
import torch
from torchvision import transforms
from PIL import Image

# ลอง Import Generator
try:
    from models2.models import Generator  
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print(f"💡 Debug: Python กำลังหา models2 ใน: {sys.path}")
    sys.exit(1)

def process_folder_to_cartoon(input_dir, output_dir):
    """
    ประมวลผลรูปภาพทั้งหมดใน input_dir โดยใช้โมเดล P2LDGAN และบันทึกผลลัพธ์ใน output_dir
    """
    
    # --- ⚙️ ตั้งค่า Path โมเดล ---
    # (ผมแก้ Path นี้ให้ตรงกับที่คุณเคยแจ้งไว้ด้วยครับ)
    model_path = "/Users/student/Desktop/research dobot/dfcall/p2ldgan_generator_200.pth"

    # ตรวจสอบว่ามีไฟล์โมเดลอยู่จริงไหม
    if not os.path.exists(model_path):
        print(f"❌ Error: ไม่พบไฟล์โมเดลที่ {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"⚙️ Using device: {device}") # ปิด print ไม่ให้รก
    
    # 1. สร้าง Generator และโหลด Checkpoint
    try:
        generator = Generator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint)
        generator.eval()
        # print("✅ Loaded checkpoint successfully.")
    except Exception as e:
        print(f"❌ Error loading model or checkpoint: {e}")
        return

    # 2. เตรียม Transformation
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # ขนาดต้องตรงกับโมเดล
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    # 3. เตรียมโฟลเดอร์ Output
    os.makedirs(output_dir, exist_ok=True)
    
    # --- เริ่มการวนลูปประมวลผลรูปภาพ ---
    
    # 4. วนลูปผ่านไฟล์ทั้งหมดในโฟลเดอร์ Input
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return

    files = os.listdir(input_dir)
    count = 0
    
    for filename in files:
        # กรองเฉพาะไฟล์รูปภาพ (jpg, jpeg, png)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            img_path = os.path.join(input_dir, filename)
            
            try:
                # โหลดรูปภาพด้วย PIL (เพื่อใช้กับ transforms)
                input_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"❌ Error opening {filename}: {e}")
                continue
                
            # 5. เตรียม Input Tensor
            input_tensor = transform(input_img).unsqueeze(0).to(device)

            # 6. Generate output
            with torch.no_grad():
                output_tensor = generator(input_tensor)
                # Denormalize: เปลี่ยน [-1, 1] เป็น [0, 1]
                output_tensor = (output_tensor * 0.5 + 0.5).clamp(0, 1)

            # 7. แปลง Tensor กลับเป็น PIL Image และบันทึก
            output_img = transforms.ToPILImage()(output_tensor.squeeze().cpu())
            
            # กำหนดชื่อไฟล์ output
            output_filename = os.path.join(output_dir, f"cartoon_{filename}")
            output_img.save(output_filename)
            count += 1
            
    print(f"✅ Done! Processed {count} images.")

# --- ตัวอย่างการเรียกใช้ฟังก์ชัน (สำหรับรันเทส) ---
if __name__ == '__main__':
    input_folder = "cropped_parts" 
    output_folder = "cartoon_output"
    process_folder_to_cartoon(input_folder, output_folder)