import cv2
import xml.dom.minidom as xmldom
import xml.etree.ElementTree as ET
import os
import numpy as np
import random


#  随机从目标视频中截取一定个数的帧
def getFrames(num, path, xml_path):
    """
     作用:随机截取一定数目的帧
     输入:视频路径
    """
    save_path = "auxiliary/task{}/img".format(num)
    os.makedirs(save_path, exist_ok=True)
    # 读取视频文件
    cap = cv2.VideoCapture(path)  # 视频的句柄
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 视频总帧数
    # 随机生成需要截取的帧,这些帧需要没有被标记
    tree = ET.parse(xml_path)
    annotations = tree.getroot()
    all_tracks = annotations.findall('track')
    all_id = [int(track.attrib['id']) for track in all_tracks]
    array = range(1, frame_count)
    # 使用列表推导式排除array2中存在的元素
    result = [x for x in array if x not in all_id]
    frames = random.sample(result, int(len(result)/200))

    for frame in frames:
        img_path = "{}/{}_0000.png".format(save_path, frame)
        print(img_path)
        # 如果还没截取这个帧,就先截取这个帧
        if not os.path.exists(img_path):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))  # frame_index要提取的帧的索引
            ret, frame = cap.read()  # ret表示图像是否读取成功 frame为读到的图像文件
            if ret:
                cv2.imwrite(img_path, frame)  # 将图像信息保存为指定格式的图像文件


def draw_outline(source_images, source_labels, output_dir):
    # 画出目标图片的物体的轮廓
    os.makedirs(output_dir, exist_ok=True)

    files = [file for file in os.listdir(source_images) if file.endswith(".png")]

    label2num = {"hem_o_lok": 1, "clip": 2, "clamp": 3, "buffer_tube": 4, "guide_needle": 5, "suture_needle": 6,
                 "syringe": 7,
                 "specimen_bag": 8, "cotton_ball": 9, "gauze": 10, "line": 11, "other_consumbles": 12,
                 "other_instrument": 13,
                 "grasping_forceps": 14, "attractor": 15, "needle_holder": 16, "scissors": 17, "ultrasonic_knife": 18,
                 "ultrasonic_clamp": 19,
                 "stapler": 20, "hook": 21}
    label2color = {}
    label_list = ["hem_o_lok", "clip", "clamp", "buffer_tube", "guide_needle", "suture_needle", "syringe",
                  "specimen_bag", "cotton_ball",
                  "gauze", "line", "other_consumbles", "other_instrument", "grasping_forceps", "attractor",
                  "needle_holder",
                  "scissors", "ultrasonic_knife", "ultrasonic_clamp", "stapler", "hook"]

    for file in files:
        print(file)
        image_path = os.path.join(source_images, file)
        file = file[0:-9]+file[-4:]
        label_path = os.path.join(source_labels, file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        for label_name in label_list:
            image = cv2.imread(image_path)
            label1 = np.zeros_like(label)
            label1[label == 1] = 255
            contours, hierarchy = cv2.findContours(label1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
            cv2.imwrite("{}/{}".format(output_dir, file), image)

            label2 = np.zeros_like(label)
            label2[label == 2] = 255
            contours, hierarchy = cv2.findContours(label2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (255, 0, 0), thickness=2)

        cv2.imwrite("{}/{}".format(output_dir, file), image)


def process_annotation(xml_path, source_labels):
    # 解析XML文件
    tree = ET.parse(xml_path)
    annotations = tree.getroot()
    all_tracks = annotations.findall('track')

    all_id = [int(track.attrib['id']) for track in all_tracks]
    if len(all_tracks) > 0:
        id = int(all_tracks[-1].attrib['id']) + 1  # 新的track的id
    else:
        id = 1
    files = [file for file in os.listdir(source_labels) if file.endswith(".png")]

    label2num = {"hem_o_lok": 1, "clip": 2, "clamp": 3, "buffer_tube": 4, "guide_needle": 5, "suture_needle": 6, "syringe": 7,
           "specimen_bag": 8, "cotton_ball": 9, "gauze": 10, "line": 11, "other_consumbles": 12, "other_instrument": 13,
           "grasping_forceps": 14, "attractor": 15, "needle_holder": 16, "scissors": 17, "ultrasonic_knife": 18, "ultrasonic_clamp": 19,
           "stapler": 20, "hook": 21}
    label_list = ["hem_o_lok", "clip", "clamp", "buffer_tube", "guide_needle", "suture_needle", "syringe", "specimen_bag", "cotton_ball",
                  "gauze", "line", "other_consumbles", "other_instrument", "grasping_forceps", "attractor", "needle_holder",
                  "scissors", "ultrasonic_knife", "ultrasonic_clamp", "stapler", "hook"]

    with open(xml_path, 'r') as file:
        lines = file.readlines()

    for file in files:
        print(file)
        label_path = os.path.join(source_labels, file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        for label_name in label_list:
            label_cpy = np.zeros_like(label)
            label_cpy[label == label2num[label_name]] = 255
            contours, hierarchy = cv2.findContours(label_cpy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                # CVAT要求至少三个点
                if len(contour) <= 2:
                    continue
                # 获取轮廓点
                contour_points = contour.reshape(-1, 2)
                # 将轮廓点转换为CVAT中的points格式
                points = ""
                for point in contour_points:
                    points += f"{point[0]:.2f},{point[1]:.2f};"
                # 删除最后一个空格
                points = points[:-1]

                new_content = '  <track id="{}" label="{}" source="semi-auto">\n'.format(id, label_name)
                id += 1
                lines.insert(-1, new_content)
                new_content = '    <polygon frame="{}" keyframe="1" outside="0" occluded="0" points="{}" z_order="0">\n'.format(file[:-4], points)
                lines.insert(-1, new_content)
                new_content = '    </polygon>\n'
                lines.insert(-1, new_content)
                new_content = '    <polygon frame="{}" keyframe="1" outside="1" occluded="0" points="{}" z_order="0">\n'.format(int(file[:-4])+1, points)
                lines.insert(-1, new_content)
                new_content = '    </polygon>\n'
                lines.insert(-1, new_content)
                new_content = '  </track>\n'
                lines.insert(-1, new_content)

    # 写回到XML文件
    # tree.write('annotations/modified_example.xml', encoding='utf-8', xml_declaration=True)
    with open('annotations/modified_example.xml', 'w') as file:
        file.writelines(lines)


if __name__ == "__main__":
    # 任务编号与相应的视频路径
    num_to_vName = {
        1: 'video/M_05202021095646_0U53469052015646_1_002_009-1modify_fps_rate.mp4',
        2: 'video/M_10102022034410_0000000015636581_2_001_0004-01',
        3: 'video/M_10102022034410_0000000015636581_2_001_0007-01modify_fps_rate.mp4',
        4: 'video/video27.mp4',
        5: 'video/UF7OtP8MljRgPAAUPiIEmodify_fps_rate.mp4',
        6: 'video/Bd7707DfvdjTjBu7enF9modify_fps_rate.mp4',
        7: 'video/JJXXoPgPY8gozkGkNqoImodify_fps_rate.mp4',
        10: 'video/05.mp4'
    }

    # 需要处理的视频文件、xml文件、任务数
    task_num = 2
    video_name = num_to_vName[task_num]
    xml_name = 'annotations/task{}.xml'.format(task_num)
    # 1.随机截取一定的帧
    # getFrames(task_num, video_name, xml_name)

    source_images = "auxiliary/task{}/img".format(task_num)
    source_labels = "auxiliary/task{}/label".format(task_num)
    output_dir = "auxiliary/task{}/pred".format(task_num)
    # 2.根据生成的标签对截取的帧进行标注
    # draw_outline(source_images, source_labels, output_dir)

    # 3.根据生成的labels处理对应的xml文件
    process_annotation(xml_name, source_labels)

