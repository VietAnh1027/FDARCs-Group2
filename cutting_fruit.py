import os
import cv2

def detect_and_crop(image, model, output_dir="cropped_fruits"):
    os.makedirs(output_dir, exist_ok=True)
    results = model(image)[0]
    fruit_list = []

    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # tọa độ
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        crop = image[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"{class_name}_{i}.jpg")
        cv2.imwrite(output_path, crop)

        fruit_list.append((class_name, crop))
        print(f"✅ Đã cắt: {class_name} tại {output_path}")

    return fruit_list
