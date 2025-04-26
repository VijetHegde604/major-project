from ultralytics import YOLO
import cv2
import os
import easyocr

# Paths
model_path = '/home/vijeth/major-project/runs/detect/run2/weights/best.pt'
test_image_dir = '/home/vijeth/major-project/AI/'
output_dir = '/home/vijeth/major-project/runs/final_output'

# Load YOLO model and OCR reader
model = YOLO(model_path)
ocr = easyocr.Reader(['en', 'kn'])  # Add more languages if needed

# Make output dir if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over test images
for img_file in os.listdir(test_image_dir):
    if not img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(test_image_dir, img_file)
    image = cv2.imread(image_path)

    results = model.predict(image_path, save=False, conf=0.5)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            cropped = image[y1:y2, x1:x2]  # Crop signboard

            # OCR
            ocr_result = ocr.readtext(cropped)
            text = ' | '.join([d[1] for d in ocr_result]) if ocr_result else "No text"
            print(text)

            # Draw box and text on original image
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Save annotated image
    save_path = os.path.join(output_dir, img_file)
    cv2.imwrite(save_path, image)
