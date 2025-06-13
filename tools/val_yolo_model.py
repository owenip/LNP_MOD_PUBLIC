from ultralytics import YOLO
import os
from pathlib import Path
import tqdm
import torch

models_fils = {
    'yolov8s': '/content/yolov8s_202503.pt',
    'yolov9s': '/content/yolov9s_202503.pt',
    'yolov10s': '/content/yolov10s_202503.pt',
    'yolov11s': '/content/yolov11s_202503.pt',
    'yolov12s-fullsize-denoised': '/content/denoise-yolov12s-fullsize.pt',
}


for model_name, model_file in tqdm.tqdm(models_fils.items()):
    print(f"Validating {model_name}...")
    print(f"Model file: {model_file}")
    model = YOLO(model_file)
    project_dir = Path('/content')
    name = model_name
    save_dir = project_dir / 'results' / name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.val(
        data=project_dir / 'data.yaml',
        imgsz=2048,
        batch=2,
        conf=0.001,
        iou=0.6,
        save=True,
        plots=True,
        save_json=True,
        project=save_dir,
    )

    torch.cuda.empty_cache()