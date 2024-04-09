# 创建相应的数据集，包括图片和对应的label
import cv2
import xml.dom.minidom as xmldom
import os
import numpy as np
import random
import shutil
from tqdm import tqdm
from datetime import date
import yaml


def mkyaml(dataset_name, id2label):
    # 定义数据
    data = {
        'path': '/home/rss/ultralytics/datasets/{}'.format(dataset_name),  # dataset root dir
        'train': 'train/images',  # train images (relative to 'path') 128 images
        'val': 'valid/images',  # val images (relative to 'path') 128 images
        'test': 'test/images',  # test images (optional)
        'names': id2label
    }

    # 指定要保存的文件路径
    file_path = 'datasets/{}/{}.yaml'.format(dataset_name, dataset_name)

    # 将数据写入YAML文件
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

    print(f"YAML文件已生成: {file_path}")


if __name__ == "__main__":

    consumables = ["hem_o_lok", "clip",  "suture_needle", "syringe", "specimen_bag",
                   "gauze", "line", "other_consumables"]
    instrument = ["forceps", "attractor", "needle_holder", "scissors", "ultrasonic_knife", "electric_hook",
                  "electrotome", "stapler", "hook", "clip_applicator", "right_angle_grab", "other_instrument"]

    # 获取当前日期
    dataset_name = "InstrumentSegment01"

    # 创建保存结果的文件夹
    os.makedirs("datasets/{}/images".format(dataset_name), exist_ok=True)
    os.makedirs("datasets/{}/labels".format(dataset_name), exist_ok=True)

    # 读取xml文件
    xml_file = xmldom.parse("annotations/annotations.xml")

    # 解决annotations中的task_id与task_num的对应,task_num与videoName的对应,以及视频帧数的问题
    meta = xml_file.getElementsByTagName('meta')[0]
    project = meta.getElementsByTagName('project')[0]
    tasks = project.getElementsByTagName('tasks')[0]
    tasks = tasks.getElementsByTagName('task')
    id2num = dict()
    id2frame = dict()  # cvat中task的id对应于计算时需要减去的帧数
    id2vName = dict()  # 任务id对应的视频名字
    id2width = dict()
    id2height = dict()
    frame_sum = 0  # 计算时需要减去的帧数
    for task in tasks:
        task_id = int(task.getElementsByTagName('id')[0].firstChild.nodeValue)
        task_name = task.getElementsByTagName('name')[0].firstChild.nodeValue
        id2num[task_id] = str(task_name[4:])  # 字符串，自己给每个视频起的任务名，如task1、task10，这里获得每个任务名中的代数
        task_size = int(task.getElementsByTagName('size')[0].firstChild.nodeValue)
        id2frame[task_id] = frame_sum
        frame_sum = frame_sum + task_size
        task_source = task.getElementsByTagName('source')[0].firstChild.nodeValue[:-4]
        id2vName[task_id] = task_source
        original_size = task.getElementsByTagName('original_size')[0]
        id2width[task_id] = float(original_size.getElementsByTagName('width')[0].firstChild.nodeValue)
        id2height[task_id] = float(original_size.getElementsByTagName('height')[0].firstChild.nodeValue)

    labels = project.getElementsByTagName('labels')[0].getElementsByTagName('label')
    label2id = dict()
    id2label = dict()
    for idx, label in enumerate(labels):
        label_name = label.getElementsByTagName('name')[0].firstChild.nodeValue
        label2id[label_name] = str(idx)
        id2label[idx] = label_name
    mkyaml(dataset_name, id2label)
    tracks = xml_file.getElementsByTagName('track')
    for track in tqdm(tracks, desc="Processing", unit="track"):
        task_id = int(track.getAttribute('task_id'))
        video_name = id2vName[task_id]
        if os.path.exists("D:/Project/cvat_related/video/{}modify_fps_rate.mp4".format(video_name)):
            cap = cv2.VideoCapture("D:/Project/cvat_related/video/{}modify_fps_rate.mp4".format(video_name))  # 视频的句柄
        else:
            cap = cv2.VideoCapture("D:/Project/cvat_related/video/{}.mp4".format(video_name))
        polygon = track.getElementsByTagName('polygon')
        if len(polygon) > 0:
            frame_num = int(polygon[0].getAttribute('frame')) - id2frame[task_id]
            frame_id = polygon[0].getAttribute('frame').rjust(6, '0')
            img_path = "datasets/{}/images/task{}_{}.png".format(dataset_name,
                                                                 id2num[task_id].rjust(2, '0'),
                                                                 frame_num.__str__().rjust(6,'0'))
            # 如果还没截取这个帧,就先截取这个帧
            if not os.path.exists(img_path):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))  # frame_index要提取的帧的索引
                ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
                if ret:
                    cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件
                else:
                    print("frame截取失败！")

            # 打开文件，如果不存在则创建，以追加模式写入
            with open("datasets/{}/labels/task{}_{}.txt".format(dataset_name,
                                                                id2num[task_id].rjust(2, '0'),
                                                                frame_num.__str__().rjust(6, '0')), 'a') as f:
                label = track.getAttribute('label')
                width = id2width[task_id]
                height = id2height[task_id]

                # 获取label位置信息
                points = polygon[0].getAttribute('points').split(";")
                out_points = np.zeros((len(points), 2))
                line = ""
                for i in range(len(points)):
                    m = points[i].split(",")
                    line = line + " {:.6g} {:.6g}".format(float(m[0])/width, float(m[1])/height)
                line += "\n"
                line = label2id[label] + line
                f.write(line)
                # if label in consumables:
                #     line = "1" + line
                #     # 写入新内容
                #     f.write(line)
                # elif label in instrument:
                #     line = "0" + line
                #     # 写入新内容
                #     f.write(line)
                # else:
                #     print("label未出现在consumables和instrument中")  # 防止在consumables和instrument两个列表中出现错误
                #     print(label)

    print("生成训练集和测试集")

    # 指定原始图片目录和输出的训练集、验证集、测试集目录
    test_ratio = 0.1
    val_ratio = 0.1
    source_images = "datasets/{}/images".format(dataset_name)
    source_labels = "datasets/{}/labels".format(dataset_name)
    output_train_dir = "datasets/{}/train".format(dataset_name)
    output_val_dir = "datasets/{}/valid".format(dataset_name)
    output_test_dir = "datasets/{}/test".format(dataset_name)
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    # 创建训练集和测试集目录
    os.makedirs(output_train_dir+"/images", exist_ok=True)
    os.makedirs(output_train_dir+"/labels", exist_ok=True)
    os.makedirs(output_val_dir+"/images", exist_ok=True)
    os.makedirs(output_val_dir+"/labels", exist_ok=True)
    os.makedirs(output_test_dir + "/images", exist_ok=True)
    os.makedirs(output_test_dir + "/labels", exist_ok=True)

    files = [file[:-4] for file in os.listdir(source_images) if file.endswith(".png")]
    random.shuffle(files)

    num_files = len(files)
    num_test = int(num_files * test_ratio)
    num_val = num_test + int(num_files * val_ratio)
    num_train = num_files - num_test

    for idx, file in enumerate(files):
        images_input_file = source_images + "/" + file + ".png"
        labels_input_file = source_labels + "/" + file + ".txt"
        if idx < num_test:
            output_images_file = output_test_dir + "/images/" + file + ".png"
            output_labels_file = output_test_dir + "/labels/" + file + ".txt"
        elif idx < num_val:
            output_images_file = output_val_dir + "/images/" + file + ".png"
            output_labels_file = output_val_dir + "/labels/" + file + ".txt"
        else:
            output_images_file = output_train_dir + "/images/" + file + ".png"
            output_labels_file = output_train_dir + "/labels/" + file + ".txt"

        shutil.copy(images_input_file, output_images_file)
        shutil.copy(labels_input_file, output_labels_file)

    os.rmdir(source_images)
    os.rmdir(source_labels)
