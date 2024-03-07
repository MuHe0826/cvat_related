import cv2
import xml.dom.minidom as xmldom
import os
import numpy as np
import random


#  随机从目标视频中截取300个帧
def getFrames(num, path):
    """
     作用:随机截取一定数目的帧
     输入:视频路径
     输出:截取的帧列表
    """
    save_path = "auxiliary/task{}/img".format(num)
    os.makedirs(save_path, exist_ok=True)
    # 读取视频文件
    cap = cv2.VideoCapture(path)  # 视频的句柄
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    frames = random.sample(range(1, frame_count), int(frame_count/200))
    for frame in frames:
        img_path = "{}/{}.png".format(save_path, frame)
        print(img_path)
        # 如果还没截取这个帧,就先截取这个帧
        if not os.path.exists(img_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))  # frame_index要提取的帧的索引
            ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
            if ret:
                cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件


if __name__ == "__main__":
    # 任务编号与相应的视频路径
    num_to_vName = {
        1: 'video/M_05202021095646_0U53469052015646_1_002_009-1modify_fps_rate.mp4',
        3: 'video/M_10102022034410_0000000015636581_2_001_0007-01modify_fps_rate.mp4',
        4: 'video/video27.mp4',
    }

    # 需要处理的视频文件、xml文件、任务数
    task_num = 4
    video_name = num_to_vName[task_num]
    xml_name = 'annotations/task{}.xml'.format(task_num)
    # 随机截取一定的帧
    getFrames(task_num, video_name)


