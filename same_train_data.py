import json
import os
import shutil
from os.path import join

import cv2
import numpy as np
import yaml
from tqdm import tqdm
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
def create_mask(source_dir,save_path):
    #从图片文件夹读取图片长和宽，然后获取txt文件生成mask

    path = os.path.join(source_dir,"images")
    if not os.path.exists(path):
        os.makedirs(path)

    txt_path = os.path.join(source_dir,"labels")
    if not os.path.exists(txt_path):
        os.makedirs(txt_path)

    save_path_images = os.path.join(save_path, "imagesTr")
    if not os.path.exists(save_path_images):
        os.makedirs(save_path_images)

    output_file_dir = os.path.join(save_path,"labelsTr")
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir, exist_ok=True)

    for filename in tqdm(os.listdir(path), desc="Processing", unit="filename"):
        file_path = os.path.join(path, filename)
        shutil.copyfile(file_path,os.path.join(save_path_images,filename[:-4]+"_0000.png"))
        txt_file = os.path.join(txt_path, filename[:-3]+"txt")
        img = cv2.imread(file_path)
        mask = np.zeros(img.shape, dtype=np.uint8)
        with open(txt_file, 'r') as file:
            # 读取文件内容
            file_content = file.readlines()
            for content in file_content:
                parts = content.split()
                label = int(parts[0])+1
                coords = list(map(float,parts[1:]))
                image_width,image_height,_ = img.shape
                pixel_coords = np.array(coords).reshape(-1, 2) * np.array([image_height, image_width])
                #mask = cv2.fillPoly(mask, np.int32([pixel_coords]), (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
                mask = cv2.fillPoly(mask, np.int32([pixel_coords]),
                                    (label,label,label))
            cv2.imwrite(os.path.join(output_file_dir, filename),mask)


def create_split_json(json_file,train_file,val_file):
    l = []
    dic = {"train":[],"val":[]}
    train_file_list = os.listdir(os.path.join(train_file,"images"))
    for filename in train_file_list:
        dic["train"].append(filename[:-4])

    val_file_list = os.listdir(os.path.join(os.path.join(val_file,"images")))
    for filename in val_file_list:
        dic["val"].append(filename[:-4])

    l.append(dic)
    with open(json_file, 'w+') as s_f_j:
        json.dump(l, s_f_j,indent=4,ensure_ascii=False)


def create_dataset_json(save_path,label_json):
    dataset_name = 'Dataset666_InstrumentSegmentation'
    num_train = len(os.listdir(os.path.join(save_path,"imagesTr")))
    with open("/home/rss/data/yolo/InstrumentSegment01.yaml", "r", encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    labels = {value:key+1 for key,value in result["names"].items()}
    labels["background"] = 0
    generate_dataset_json(save_path, {0: 'R', 1: 'G', 2: 'B'},
                          labels,
                          num_train, '.png', dataset_name=dataset_name)

if __name__ == '__main__':
    train_file = '/home/rss/data/yolo/InstrumentSegment01/train'
    save_path = "/home/zj/nnUNet/nnUNetFrame/DATASET/nnUNet_raw/Dataset666_InstrumentSegmentation"  #nnunet训练集目录
    valid_file = '/home/rss/data/yolo/InstrumentSegment01/valid'
    test_file = '/home/rss/data/yolo/InstrumentSegment01/test'  #yolo测试集目录
    json_file = "/home/zj/nnUNet/nnUNetFrame/DATASET/nnUNet_preprocessed/Dataset666_InstrumentSegmentation/splits_final.json" #更改nnunet指定训练、验证集
    label_json = "/home/rss/data/yolo/InstrumentSegment01.yaml"  #存放yolo标签数据文件
    # create_mask(valid_file,save_path)
    # create_mask(train_file,save_path)
    create_dataset_json(save_path,label_json)
    create_split_json(json_file,train_file,valid_file)