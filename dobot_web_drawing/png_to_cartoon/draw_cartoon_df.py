import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
from models2.models import Generator  # อย่าลืม: Generator ต้องอยู่ใน path ที่ถูกต้อง

def process_folder_to_cartoon(input_dir, output_dir):
    """
    ประมวลผลรูปภาพทั้งหมดใน input_dir โดยใช้โมเดล P2LDGAN และบันทึกผลลัพธ์ใน output_dir
    """
    
    # --- การตั้งค่าโมเดล (โหลดครั้งเดียว) ---
    model_path = r"E:\P2LDGAN\p2ldgan_generator_200.pth"  # ใส่ path checkpoint ของคุณ

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. สร้าง Generator และโหลด Checkpoint
    try:
        generator = Generator().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        generator.load_state_dict(checkpoint)
        generator.eval()
        print("✅ Loaded checkpoint successfully.")
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
    print(f"📁 Output will be saved to: {output_dir}")

    # --- เริ่มการวนลูปประมวลผลรูปภาพ ---
    
    # 4. วนลูปผ่านไฟล์ทั้งหมดในโฟลเดอร์ Input
    for filename in os.listdir(input_dir):
        # กรองเฉพาะไฟล์รูปภาพ (jpg, jpeg, png)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            
            img_path = os.path.join(input_dir, filename)
            print(f"\n🖼️ Processing: {filename}")
            
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
            print(f"✅ Generated image saved as {output_filename}")


# --- ตัวอย่างการเรียกใช้ฟังก์ชัน ---

if __name__ == '__main__':
    # กำหนดโฟลเดอร์ที่คุณต้องการประมวลผล
    input_folder = "cropped_parts"  # โฟลเดอร์ที่มีรูป 256x256 ที่ถูก crop
    output_folder = "cartoon_output" # โฟลเดอร์สำหรับผลลัพธ์

    process_folder_to_cartoon(input_folder, output_folder)
