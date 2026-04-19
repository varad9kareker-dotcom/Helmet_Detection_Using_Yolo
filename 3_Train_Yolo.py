import time
from ultralytics import YOLO

start_time = None

def on_train_start(trainer):
    global start_time
    start_time = time.time()

def on_train_epoch_end(trainer):
    elapsed = time.time() - start_time
    epoch = trainer.epoch + 1
    total_epochs = trainer.args.epochs

    avg_time = elapsed / epoch
    remaining_epochs = total_epochs - epoch
    eta = avg_time * remaining_epochs

    hrs  = int(eta // 3600)
    mins = int((eta % 3600) // 60)
    secs = int(eta % 60)

    print(f"⏳ ETA: {hrs:02d}:{mins:02d}:{secs:02d}")

def main():
    model = YOLO('yolo11n.pt')  # and update name too

    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)

    model.train(
        data='D:/Helmet_Detection/Helmet_Detection_YOLO/dataset.yaml',  # ← fixed
        epochs=50,
        imgsz=640,
        batch=6,        # ← lowered from 8 (CPU has no VRAM limit but RAM is tight)
        name='helmet_VK1',
        device='0',   # ← fixed, you have no GPU
        patience=20,
        workers=2,      # ← set to 0 on Windows CPU to avoid multiprocessing issues
        pretrained=True,
        verbose=True
    )

if __name__ == "__main__":
    main()