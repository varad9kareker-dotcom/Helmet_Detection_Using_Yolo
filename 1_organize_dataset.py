# organize_dataset.py
import shutil
import random
import xml.etree.ElementTree as ET
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────
base        = Path(r'D:\Helmet_Detection')
img_src     = base / 'Images'
ann_src     = base / 'annotations'
out         = base / 'Helmet_Detection_YOLO'
SPLIT_RATIO = 0.8   # 80% train, 20% val
SEED        = 42

# Exact strings from your <n> tag
CLASS_MAP = {
    'With Helmet':    0,
    'Without Helmet': 1,
}
# ──────────────────────────────────────────────────────────────────────────────

def convert_xml_to_yolo(xml_path: Path, class_map: dict) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size  = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)

    lines = []
    for obj in root.findall('object'):
        # Handle both <n> and <name> tags
        name_tag = obj.find('n')
        if name_tag is None:
            name_tag = obj.find('name')
        if name_tag is None:
            print(f"  [SKIP] No class tag found in {xml_path.name}")
            continue

        cls_name = name_tag.text.strip()
        if cls_name not in class_map:
            print(f"  [SKIP] Unknown class '{cls_name}' in {xml_path.name}")
            continue

        cls_id = class_map[cls_name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)

        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h

        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def main():
    # Gather all images that have a matching annotation
    img_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [
        p for p in img_src.iterdir()
        if p.suffix.lower() in img_extensions
        and (ann_src / p.with_suffix('.xml').name).exists()
    ]

    if not all_images:
        print("No matched image+annotation pairs found. Check your paths.")
        return

    print(f"Found {len(all_images)} matched image+annotation pairs.")

    # Shuffle and split
    random.seed(SEED)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * SPLIT_RATIO)
    splits = {
        'train': all_images[:split_idx],
        'val':   all_images[split_idx:],
    }

    # Create output dirs
    for split_name in splits:
        (out / 'images' / split_name).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split_name).mkdir(parents=True, exist_ok=True)

    # Process each split
    total_boxes = 0
    for split_name, images in splits.items():
        img_out = out / 'images' / split_name
        lbl_out = out / 'labels' / split_name
        skipped = 0

        for img_path in images:
            xml_path = ann_src / img_path.with_suffix('.xml').name

            yolo_lines = convert_xml_to_yolo(xml_path, CLASS_MAP)
            if not yolo_lines:
                skipped += 1
                continue

            shutil.copy(img_path, img_out / img_path.name)

            lbl_path = lbl_out / img_path.with_suffix('.txt').name
            lbl_path.write_text('\n'.join(yolo_lines))
            total_boxes += len(yolo_lines)

        print(f"{split_name:>5}: {len(images) - skipped} images | {skipped} skipped")

    print(f"\nTotal boxes written : {total_boxes}")
    print(f"Output ready at     : {out}")

    # Write dataset.yaml
    yaml_content = f"""\
path: {out.as_posix()}
train: images/train
val:   images/val

nc: {len(CLASS_MAP)}
names: {list(CLASS_MAP.keys())}
"""
    (out / 'dataset.yaml').write_text(yaml_content)
    print("dataset.yaml written.")


if __name__ == '__main__':
    main()