from ultralytics import YOLO
import cv2
import os

image_path = 'fruits.jpg'
model_path = 'model/best.pt'
output_dir = 'cropped_fruits'

# LƯU Ý: ĐÂY MỚI CHỈ LÀ MODEL DEMO, CHƯA PHẢI MODEL CHÍNH THỨC
model = YOLO(model_path)
image = cv2.imread(image_path)

results = model(image)[0]

for i, box in enumerate(results.boxes):
    # Lấy toạ độ bounding box
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Toạ độ góc trái trên và phải dưới
    
    # Lấy tên lớp (label)
    cls_id = int(box.cls[0])
    class_name = model.names[cls_id]

    # Cắt vùng ảnh
    fruit_crop = image[y1:y2, x1:x2]

    # Lưu ảnh cắt
    output_path = os.path.join(output_dir, f"{class_name}_{i}.jpg")
    cv2.imwrite(output_path, fruit_crop)

    print(f"Đã lưu {output_path}")

print("✅ Hoàn tất cắt ảnh.")
