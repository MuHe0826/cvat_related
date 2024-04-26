import numpy as np
import matplotlib.pyplot as plt
import os
# 标签掩码数据
import cv2
from ultralytics import YOLO
import random
def make_label_mask(path, filename,txt_path,save_path):
    #从图片文件夹读取图片长和宽，然后获取txt文件生成mask
    file_path = os.path.join(path, filename)
    output_file_dir = os.path.join(save_path,"label")
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir, exist_ok=True)
    txt_file = os.path.join(txt_path, filename[:-3]+"txt")
    img = cv2.imread(file_path)
    mask = np.zeros(img.shape, dtype=np.uint8)
    with open(txt_file, 'r') as file:
        # 读取文件内容
        file_content = file.readlines()
        for content in file_content:
            parts = content.split()
            label = int(parts[0])
            coords = list(map(float,parts[1:]))
            image_width,image_height,_ = img.shape
            pixel_coords = np.array(coords).reshape(-1, 2) * np.array([image_height, image_width])
            #mask = cv2.fillPoly(mask, np.int32([pixel_coords]), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            mask = cv2.fillPoly(mask, np.int32([pixel_coords]),
                                (label,label,label))
        cv2.imwrite(os.path.join(output_file_dir, filename),mask)

def make_oredict_mask(path,save_path,pt_path):
    model = YOLO(pt_path)
    output_file_dir = os.path.join(save_path, "predict")
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir, exist_ok=True)
    for image_name in os.listdir(path):
        img_path = os.path.join(path, image_name)
        img = cv2.imread(img_path)
        results = model(img)
        for i,det in enumerate(results[0].masks.xy):
            label = int(results[0].boxes.cls[i])
            mask = np.zeros(img.shape, dtype=np.uint8)
            pixel_coords=det
            try:
                mask = cv2.fillPoly(mask, np.int32([pixel_coords]),
                                    (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            except Exception as e:
                print("mask_error++++++++++++++++++++++++++++++++++++++++++==")
            #mask = cv2.fillPoly(mask, np.int32([pixel_coords]),(label, label, label))
        cv2.imwrite(os.path.join(output_file_dir, image_name), mask)

if __name__ == "__main__":
    path = r'C:\Users\jozon\Desktop\valid\images'
    txt_path = r'C:\Users\jozon\Desktop\valid\labels'
    save_path =r"C:\Users\jozon\Desktop\valid\mask"
    pt_path = r"C:\Users\jozon\Downloads\yolo\v8n\yolov8n-seg.pt"
    os.makedirs(save_path, exist_ok=True)
    # for filename in os.listdir(path):
    #     if filename.endswith('.png'):
    #         make_label_mask(path, filename,txt_path,save_path)
    make_oredict_mask(path, save_path,pt_path)
