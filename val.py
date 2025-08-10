import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('D:/工程创新/最终训练/ultralytics-main/ultralytics-main/runs/train/exp8/weights/best.pt')
    model.val(data='D:/工程创新/最终训练/ultralytics-main/ultralytics-main/breast_tumor01/data.yaml',
              split='val',
              imgsz=640,
              batch=4,
              iou=0.6,
              rect=False,
              save_json=False,
              project='runs/val',
              name='exp',
              )