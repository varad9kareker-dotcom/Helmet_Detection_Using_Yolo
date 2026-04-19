# verify_clean.py
import cv2
import random
from pathlib import Path

img_dir = Path('D:\\Helmet_Detection\\Helmet_Detection_YOLO\\images\\train')
lbl_dir = Path('D:\\Helmet_Detection\\Helmet_Detection_YOLO\\labels\\train')

CLASS_NAMES = ['With Helmet', 'Without Helmet']
COLORS = [
    (255,  80,  80),   # With Helmet - red
    ( 80, 255,  80),   # Without Helmet - green
]

def draw_boxes(img_path, lbl_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    H, W = img.shape[:2]

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if lbl_path.exists():
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])
                x1 = int((xc - bw/2) * W)
                y1 = int((yc - bh/2) * H)
                x2 = int((xc + bw/2) * W)
                y2 = int((yc + bh/2) * H)
                color = COLORS[cls % len(COLORS)]
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
                cv2.putText(img, CLASS_NAMES[cls], (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img

# Show 6 random samples
samples = random.sample(list(img_dir.glob('*.png')), 6)
for img_path in samples:
    lbl_path = lbl_dir / (img_path.stem + '.txt')
    result   = draw_boxes(img_path, lbl_path)
    if result is not None:
        result = cv2.resize(result, (640, 512))
        cv2.imshow(img_path.name, result)

cv2.waitKey(0)
cv2.destroyAllWindows()