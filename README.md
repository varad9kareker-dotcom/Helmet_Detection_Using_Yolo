Helmet Detection
Real-Time Safety Helmet Detection using YOLO11n
Trained on custom dataset  |  Edge-ready for NVIDIA Jetson


Overview
This project implements a real-time helmet detection system using YOLO11n, the latest and most lightweight model from Ultralytics. The system is designed to detect whether individuals on bikes or motorcycles are wearing helmets, making it suitable for traffic safety monitoring applications.
The model is optimized for deployment on edge hardware, particularly NVIDIA Jetson platforms, while being trained on a Windows PC with an NVIDIA GTX 1050 Ti GPU.

Detection Classes

Class ID	Class Name	Description
0	With Helmet	Person wearing a safety helmet
1	Without Helmet	Person not wearing a safety helmet

Dataset
Source Structure
The raw dataset consists of images with Pascal VOC-style XML annotations (non-standard, using <n> tag instead of <name>):
D:\Helmet_Detection\
    Images\          <- source images (.png)
    annotations\     <- per-image XML files

Prepared Dataset (YOLO Format)
Helmet_Detection_YOLO\
    images\
        train\       <- 609 images (80%)
        val\         <- 152 images (20%)
    labels\
        train\       <- 609 .txt label files
        val\         <- 152 .txt label files
    dataset.yaml

Dataset Statistics

Metric	Value
Total image-annotation pairs	764
Training images	609 (80%)
Validation images	152 (20%)
Skipped (empty/corrupt)	3
Total bounding boxes	1,451
Annotation format (source)	Pascal VOC XML (non-standard <n> tag)
Label format (YOLO)	Normalized cx cy w h per line

Data Preprocessing
organize_dataset.py
Converts raw XML annotations to YOLO format and performs an 80/20 train/val split. Key steps:
•	Matches each image to its corresponding XML annotation file
•	Parses both <n> and <name> XML tags to handle mixed annotation formats
•	Converts Pascal VOC bounding boxes (xmin ymin xmax ymax) to YOLO format (cx cy w h), normalized by image dimensions
•	Randomly shuffles and splits dataset (seed=42 for reproducibility)
•	Auto-generates dataset.yaml for YOLO training

verify_clean.py
Visual sanity-check script that overlays predicted bounding boxes on 6 random training images using OpenCV. Confirms annotation conversion is spatially correct before committing to a full training run.

Model

Property	Value
Architecture	YOLO11n (Ultralytics)
Parameters	2.6M
Model size	~5.4 MB
Input size	640 x 640
Pretrained weights	yolo11n.pt (COCO)
Task	Object Detection

YOLO11n was chosen for its minimal footprint while maintaining competitive accuracy, making it ideal for deployment on NVIDIA Jetson edge devices.

Training
Hardware
•	GPU: NVIDIA GeForce GTX 1050 Ti (4GB VRAM)
•	Training platform: Windows PC, Visual Studio 2022
•	Framework: Ultralytics YOLOv8/YOLO11 Python API

Hyperparameters

Parameter	Value	Notes
epochs	100	With early stopping (patience=20)
imgsz	640	Standard YOLO input size
batch	8	Fits within 4GB VRAM
device	0	CUDA GPU
workers	2	Windows-safe with NVIDIA GPU
optimizer	auto	Ultralytics default (AdamW)
pretrained	True	Fine-tuned from COCO weights

Training Script
python 2_train.py

A custom ETA callback prints estimated time remaining after each epoch, making long CPU/GPU runs easier to monitor.

Inference & Deployment
Python Inference
from ultralytics import YOLO
model = YOLO('runs/detect/helmet_yolo11n_v1/weights/best.pt')
results = model('image.png')

Export for NVIDIA Jetson (TensorRT)
After training, export to TensorRT for maximum inference speed on Jetson:
model.export(format='engine', device=0)

Platform	Format	Expected FPS
NVIDIA Jetson Orin	TensorRT (.engine)	60+ FPS
GTX 1050 Ti (PC)	PyTorch (.pt)	~30 FPS
CPU (fallback)	ONNX (.onnx)	~5 FPS

Requirements
Python Packages
•	ultralytics
•	opencv-python
•	torch + torchvision (CUDA build for GPU training)

Install
pip install ultralytics opencv-python
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Verify GPU
python -c "import torch; print(torch.cuda.is_available())"

Project Structure
D:\Helmet_Detection\
    Images\                    <- raw images
    annotations\               <- raw XML annotations
    Helmet_Detection_YOLO\     <- prepared YOLO dataset
    1_organize_dataset.py      <- XML -> YOLO conversion + split
    2_verify_clean.py          <- visual bbox sanity check
    3_train.py                 <- YOLO11n training script
    runs\detect\              <- training outputs (auto-generated)


Built with Ultralytics YOLO11  |  OpenCV  |  PyTorch
