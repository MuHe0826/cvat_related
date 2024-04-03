# 根据annotation文件，从目标视频中截取相应的帧
import cv2
import xml.dom.minidom as xmldom
import os


# 需要处理的视频文件、xml文件、任务数
video_name = '../video/M_05202021095646_0U53469052015646_1_002_009-1modify_fps_rate.mp4'
task_num = 3
xml_name = 'annotations.xml'
# 创建保存结果的文件夹
save_path = "dataset/images"
if not os.path.exists(save_path):
    os.makedirs(save_path)


# 读取xml文件
xml_file = xmldom.parse(xml_name)
# 读取视频文件
cap = cv2.VideoCapture(video_name)  # 视频的句柄
fps = cap.get(cv2.CAP_PROP_FPS)  # 视频的帧率
print('视频的帧率为{}'.format(fps))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
print('视频的总帧数为{}'.format(frame_count))
# 获取xml文件中的元
tracks = xml_file.getElementsByTagName('track')

for track in tracks:
    polygon = track.getElementsByTagName('polygon')
    if len(polygon) > 0:
        frame_num = polygon[0].getAttribute('frame')
        print("正在处理第{}帧...".format(frame_num))
        frame_num_frame = frame_num.rjust(6,'0')
        img_path = "{}/frame_{}.png".format(save_path,    frame_num_frame)
        # 如果还没截取这个帧,就先截取这个帧
        if not os.path.exists(img_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_num))  # frame_index要提取的帧的索引
            ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
            if ret:
                cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件

