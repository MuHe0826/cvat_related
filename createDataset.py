# 创建相应的数据集，包括图片和对应的label
import cv2
import xml.dom.minidom as xmldom
import os
import numpy as np
import random
import shutil
from tqdm import tqdm


consumables = ["hem_o_lok", "clip", "clamp", "buffer_tube", "guide_needle", "suture_needle", "syringe", "specimen_bag",
               "cotton_ball", "gauze", "line", "other_consumables"]
instrument = ["grasping_forceps", "attractor", "needle_holder", "scissors", "ultrasonic_knife", "ultrasonic_clamp",
              "stapler", "hook", "other_instrument"]

# 创建保存结果的文件夹
os.makedirs("dataset/images", exist_ok=True)
os.makedirs("dataset/labels", exist_ok=True)

# 读取xml文件
xml_file = xmldom.parse("annotations.xml")

# 解决annotations中的task_id与task_num的对应,task_num与videoName的对应,以及视频帧数的问题
meta = xml_file.getElementsByTagName('meta')[0]
project = meta.getElementsByTagName('project')[0]
tasks = project.getElementsByTagName('tasks')[0]
tasks = tasks.getElementsByTagName('task')
id2num = dict()
id2frame = dict()  # cvat中task的id对应于计算时需要减去的帧数
num_to_vName = dict()
frame_sum = 0  # 计算时需要减去的帧数
for task in tasks:
    task_id = int(task.getElementsByTagName('id')[0].firstChild.nodeValue)
    task_name = task.getElementsByTagName('name')[0].firstChild.nodeValue
    task_num = int(task_name[4:])
    id2num[task_id] = task_num
    task_size = int(task.getElementsByTagName('size')[0].firstChild.nodeValue)
    id2frame[task_id] = frame_sum
    frame_sum = frame_sum + task_size
    task_source = task.getElementsByTagName('source')[0].firstChild.nodeValue[:-4]
    num_to_vName[task_num] = task_source

tracks = xml_file.getElementsByTagName('track')
for track in tqdm(tracks, desc="Processing", unit="track"):
    task_id = int(track.getAttribute('task_id'))
    task_num = id2num[task_id]
    video_name = num_to_vName[task_num]
    if os.path.exists("video/{}modify_fps_rate.mp4".format(video_name)):
        cap = cv2.VideoCapture("video/{}modify_fps_rate.mp4".format(video_name))  # 视频的句柄
    else:
        cap = cv2.VideoCapture("video/{}.mp4".format(video_name))
    polygon = track.getElementsByTagName('polygon')
    if len(polygon) > 0:
        frame_num = int(polygon[0].getAttribute('frame')) - id2frame[task_id]
        img_path = "dataset/images/task{}_{}.png".format(task_num, frame_num)
        # 如果还没截取这个帧,就先截取这个帧
        if not os.path.exists(img_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))  # frame_index要提取的帧的索引
            ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
            if ret:
                cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件
        img = cv2.imread(img_path)
        label = track.getAttribute('label')
        label_path = "./dataset/labels/task{}_{}.png".format(task_num, frame_num)
        # 如果还没创建相应的mask文件,就先创建mask文件
        if not os.path.exists(label_path):
            img = np.zeros(img.shape, dtype=np.uint8)  # 这里的大小为图片大小 1920 x 1080
            cv2.imwrite(label_path, img)
        # 获取label位置信息
        points = polygon[0].getAttribute('points').split(";")
        out_points = np.zeros((len(points), 2))
        for i in range(len(points)):
            m = points[i].split(",")
            out_points[i][0] = float(m[0])
            out_points[i][1] = float(m[1])
        # 标注mask并保存
        # img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # 以黑白格式读取图片
        img = cv2.imread(label_path)
        if label in consumables:
            img_fill = cv2.fillPoly(img, np.int32([out_points]), (1, 1, 1))
            cv2.imwrite(label_path, img_fill)
        elif label in instrument:
            img_fill = cv2.fillPoly(img, np.int32([out_points]), (2, 2, 2))
            cv2.imwrite(label_path, img_fill)
        else:
            print("error")  # 防止在consumables和instrument两个列表中出现错误
            print(label)


print("生成训练集和测试集")

# 指定原始图片目录和输出的训练集、测试集目录
test_ratio = 0.2
source_images = "dataset/images"
source_labels = "dataset/labels"
output_train_dir = "dataset/train"
output_test_dir = "dataset/test"

# 创建训练集和测试集目录
os.makedirs(output_train_dir+"/images", exist_ok=True)
os.makedirs(output_train_dir+"/labels", exist_ok=True)
os.makedirs(output_test_dir+"/images", exist_ok=True)
os.makedirs(output_test_dir+"/labels", exist_ok=True)

files = [file for file in os.listdir(source_images) if file.endswith(".png")]
random.shuffle(files)

num_files = len(files)
num_test = int(num_files * test_ratio)
num_train = num_files - num_test

for idx, file in enumerate(files):
    images_input_file = os.path.join(source_images, file)
    labels_input_file = os.path.join(source_labels, file)
    if idx < num_test:
        output_images_file = os.path.join(output_test_dir + "/images", file)
        output_labels_file = os.path.join(output_test_dir + "/labels", file)
    else:
        output_images_file = os.path.join(output_train_dir + "/images", file)
        output_labels_file = os.path.join(output_train_dir + "/labels", file)

    shutil.copy(images_input_file, output_images_file)
    shutil.copy(labels_input_file, output_labels_file)
