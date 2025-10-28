import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    #model = YOLO('/root/ultralytics/ultralytics/cfg/models/11/yolo11s-oper.yaml')
    #model = YOLO('/kaggle/working/ultralytics/ultralytics/cfg/models/11/yolo11s-oper.yaml')
    model = YOLO('/kaggle/working/ultralytics/ultralytics/cfg/models/11/yolo11s-oper04.yaml')
    #model = YOLO('/content/ultralytics/ultralytics/cfg/models/11/yolo11-oper.yaml')
    #model = YOLO('/kaggle/input/tumor-yolo/finally-best.pt')
    #model = YOLO('/kaggle/input/tumor-yolo/tumor-12.pt')
    #model.load('/content/drive/MyDrive/KaggleNotebookOutput/ultralytics/runs/train/exp/weights/best.pt') # loading pretrain weights
    #model.load('/kaggle/input/tumor-yolo/finally-best.pt') # loading pretrain weights
    model.train(#data='/root/.cache/kagglehub/datasets/monaerkiconbinker/tumor-yolo/versions/2/breast_tumor04/data.yaml',
                data='/kaggle/input/tumor-yolo/breast_tumor05/breast_tumor05/data.yaml',
                #data='/content/datasets/breast_tumor04/data.yaml'
                cache=False,
                imgsz=640,
                epochs=500,
                batch= 32,
                close_mosaic=0,
                device='0,1',
                optimizer='AdamW', # using BGD
                lr0= 0.001, # 较小的初始学习率
                lrf=0.01,
                cos_lr=True,
                momentum= 0.9,
                #betas: [0.9, 0.999]  # 这是AdamW的"动量"参数
                weight_decay=0.01,
                project='runs/train',
                name='exp',
                patience=150,
                workers=0,
                val=True,
                amp=False,
                #save_period = 20,
                #python train.py --img 640 --batch 16 --epochs 400 --data your_dataset.yaml --weights yolov8n.pt --patience 100 --close_mosaic 0 --name Stage1_Mosaic_Training
                #python train.py --img 640 --batch 16 --epochs 50 --data your_dataset.yaml --weights runs/train/Stage1_Mosaic_Training/weights/best.pt --patience 30 --lr0 0.001 --no-aug --name Stage2_Finetuning
                )
