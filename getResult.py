# 根据annotation文件，从目标视频中截取相应的帧
import cv2
import xml.dom.minidom as xmldom
import os
import numpy as np
from tqdm import tqdm

# 创建保存结果的文件夹
os.makedirs("result", exist_ok=True)

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

# 获取label的名称与颜色的对应关系
labels = project.getElementsByTagName('labels')[0].getElementsByTagName('label')
dic = dict()
for label in labels:
    label_name = label.getElementsByTagName('name')[0].firstChild.nodeValue  # !!!这里不知道为何
    label_color = label.getElementsByTagName('color')[0].firstChild.nodeValue
    # 将十六进制颜色值转换为RGB值
    r = int(label_color[1:3], 16)
    g = int(label_color[3:5], 16)
    b = int(label_color[5:7], 16)
    dic[label_name] = (b, g, r) # cv2.polylines()函数输入的颜色是（B,G,R）格式


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
        label = track.getAttribute('label')
        img_path = "result/task{}_{}.png".format(task_num, frame_num)
        # 如果还没截取这个帧,就先截取这个帧
        if not os.path.exists(img_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))  # frame_index要提取的帧的索引
            ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
            if ret:
                cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件
        # 获取label位置信息
        points = polygon[0].getAttribute('points').split(";")
        out_points = np.zeros((len(points), 2))
        for i in range(len(points)):
            m = points[i].split(",")
            out_points[i][0] = float(m[0])
            out_points[i][1] = float(m[1])
        pts = out_points.reshape((-1, 1, 2))
        # 标注label并保存
        img = cv2.imread(img_path)
        img_poly = cv2.polylines(img, np.int32([pts]), isClosed=True, color=dic[label], thickness=2)
        cv2.imwrite(img_path, img_poly)
