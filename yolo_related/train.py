from ultralytics import YOLO

# Load a model
# model = YOLO("datasets/yolov8-seg.yaml")  # build a new model from scratch
# model = YOLO('yolov8n-seg.pt')  # load a pretrained model (recommended for training)
model = YOLO('/home/zj/ultralytics/datasets/yolov8-seg.yaml').load('/home/zj/ultralytics/datasets/yolov8n-seg.pt')  # build from YAML and transfer weights

# Use the model
model.train(data="/home/zj/ultralytics/datasets/coco128-seg.yaml", task="segment",mode="train",workers=0,batch=3,epochs=300,device=[0,2,4])  # train the model