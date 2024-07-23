# 创建相应的数据集，包括图片和对应的label
import xml.dom.minidom as xmldom
import os
from tqdm import tqdm
import xlsxwriter as xw
def creat_excel(name2num,excel_save_path):
    excel_name = os.path.join(excel_save_path,"各视频类别计数.xlsx")
    workbook = xw.Workbook(excel_name)  # 创建工作簿
    for key, value in name2num.items():
        worksheet1 = workbook.add_worksheet(key[:20]+key[-10:])  # 创建子表
        worksheet1.activate()  # 激活表
        title = ['类别', '数量']  # 设置表头
        worksheet1.write_row("A1", title)
        row_num = 1
        for key, value in value.items():
            row_num += 1
            row = 'A' + str(row_num)
            insert_data = [key, value]
            worksheet1.write_row(row, insert_data)
    workbook.close()

def parse_xml(xml_name):
    xml_file = xmldom.parse(xml_name)
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
    # 统计每个视频的类别数量
    name2num = dict()
    video_names = []
    frame_sum = 0  # 计算时需要减去的帧数
    for task in tasks:
        task_id = int(task.getElementsByTagName('id')[0].firstChild.nodeValue)
        task_name = task.getElementsByTagName('name')[0].firstChild.nodeValue
        id2num[task_id] = str(task_name[4:])  # 字符串，自己给每个视频起的任务名，如task1、task10，这里获得每个任务名中的代数
        task_size = int(task.getElementsByTagName('size')[0].firstChild.nodeValue)
        id2frame[task_id] = frame_sum
        frame_sum = frame_sum + task_size
        task_source = task.getElementsByTagName('source')[0].firstChild.nodeValue[:-4]
        video_names.append(task_source)
        id2vName[task_id] = task_source
        original_size = task.getElementsByTagName('original_size')[0]
        id2width[task_id] = float(original_size.getElementsByTagName('width')[0].firstChild.nodeValue)
        id2height[task_id] = float(original_size.getElementsByTagName('height')[0].firstChild.nodeValue)

    labels = project.getElementsByTagName('labels')[0].getElementsByTagName('label')
    cls2num = dict()
    label2id = dict()
    id2label = dict()
    for idx, label in enumerate(labels):
        label_name = label.getElementsByTagName('name')[0].firstChild.nodeValue
        cls2num[label_name] = 0
        label2id[label_name] = str(idx)
        id2label[idx] = label_name
    for video_name in video_names:
        name2num[video_name] = cls2num.copy()
    tracks = xml_file.getElementsByTagName('track')
    for track in tqdm(tracks, desc="Processing", unit="track"):
        task_id = int(track.getAttribute('task_id'))
        video_name = id2vName[task_id]
        polygon = track.getElementsByTagName('polygon')
        if len(polygon) > 0:
            frame_num = int(polygon[0].getAttribute('frame')) - id2frame[task_id]
            frame_id = polygon[0].getAttribute('frame').rjust(6, '0')
            label = track.getAttribute('label')
            name2num[video_name][label] += 1
            print(video_name, label, name2num[video_name][label])
    return name2num

if __name__ == "__main__":
    xml_name = r"D:\pycharm\pythonProject1\utils\yolo_related\annotations.xml"
    excel_save_path = r"D:\pycharm\pythonProject1\utils\yolo_related"
    name2num = parse_xml(xml_name)  # 读取xml文件
    creat_excel(name2num,excel_save_path)   # 写进excel


