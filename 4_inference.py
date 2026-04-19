# inference_flir.py
import cv2
from pathlib import Path
from ultralytics import YOLO

model = YOLO('D:\\Helmet_Detection\\runs\\detect\\helmet_VK1\\weights\\best.pt')

CLASS_NAMES = ['With Helmet', 'Without Helmet']
COLORS = [
    (255,  80,  80),   # With Helmet - red
    ( 80, 255,  80),   # Without Helmet - green
]

def run_inference(img_path, conf=0.4):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Cannot load: {img_path}")
        return

    results = model.predict(img, conf=conf, verbose=False)
    boxes   = results[0].boxes

    # Draw manually for full control
    counts = {name: 0 for name in CLASS_NAMES}

    for box in boxes:
        cls  = int(box.cls)
        conf_val = float(box.conf)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS[cls % len(COLORS)]
        label = f"{CLASS_NAMES[cls]} {conf_val:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(img, (x1, y1-22), (x1+len(label)*11, y1), color, -1)
        cv2.putText(img, label, (x1+2, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        counts[CLASS_NAMES[cls]] += 1

    # Count summary in top-left
    y_pos = 25
    for name, count in counts.items():
        if count > 0:
            cv2.putText(img, f"{name}: {count}", (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            y_pos += 25

    return img


# ── Run on val images ─────────────────────────────────────────────────
val_dir = Path('D:\\Helmet_Detection\\Helmet_Detection_YOLO\\images\\val')
images  = list(val_dir.glob('*.png'))

print(f"Found {len(images)} val images. Press any key to advance, Q to quit.")

for img_path in images:
    result = run_inference(img_path, conf=0.4)
    if result is not None:
        cv2.imshow('Helmet Detection', result)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break

cv2.destroyAllWindows()