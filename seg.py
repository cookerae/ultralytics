import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/工程创新/最终训练/ultralytics-main/ultralytics-main/ultralytics/cfg/models/11/yolo11.yaml')
    #model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='D:/工程创新/最终训练/ultralytics-main/ultralytics-main/breast_tumor04/data.yaml',
                cache=False,
                imgsz=640,
                epochs=20,
                batch= 4,
                close_mosaic=10,
                device='0',
                optimizer='AdamW', # using BGD
                project='runs/train',
                name='exp',
                patience=10,
                workers=0,
                val=True,
                amp=True,
                save_period = 5,
                #python train.py --img 640 --batch 16 --epochs 400 --data your_dataset.yaml --weights yolov8n.pt --patience 100 --close_mosaic 0 --name Stage1_Mosaic_Training
                #python train.py --img 640 --batch 16 --epochs 50 --data your_dataset.yaml --weights runs/train/Stage1_Mosaic_Training/weights/best.pt --patience 30 --lr0 0.001 --no-aug --name Stage2_Finetuning
                )