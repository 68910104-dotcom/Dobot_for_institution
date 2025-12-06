from rembg import remove
from PIL import Image

def rmbg_image(input_path):
    output_path = 'output_with_white_bg.png' # <-- ชื่อไฟล์ผลลัพธ์

    # 2. เปิดรูปภาพต้นฉบับ
    try:
        input_image = Image.open(input_path)
    except FileNotFoundError:
        print(f"Error: ไม่พบไฟล์ '{input_path}'")
        exit()

    # 3. ลบพื้นหลัง และแทนที่ด้วยสีขาว
    # bgcolor=(255, 255, 255, 255) คือค่าสีขาว (R, G, B, Alpha)
    output_image = remove(input_image, bgcolor=(255, 255, 255, 255))

    # 4. บันทึกรูปภาพผลลัพธ์
    output_image.save(output_path)

    print(f"ลบพื้นหลังและแทนที่ด้วยสีขาวเรียบร้อย! บันทึกไฟล์แล้วที่: {output_path}")

    # (ไม่จำเป็น) แสดงรูปภาพผลลัพธ์ (ถ้าคุณรันในสภาพแวดล้อมที่แสดงผลได้)
    # output_image.show()