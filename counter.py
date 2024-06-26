import os
import xml.dom.minidom as xmldom
import xlsxwriter as xw
import time

from alive_progress import alive_bar

# dic = {"hem_o_lok": 0, "clip": 0, "clamp":0, "buffer_tube":0, "guide_needle":0, "suture_needle":0,  "specimen_bag":0,
#        "cotton_ball":0, "gauze":0, "line":0, "grasping_forceps":0, "attractor":0, "needle_holder":0,
#        "scissors":0, "ultrasonic_knife":0, "ultrasonic_clamp":0, "electrotome":0 ,
#               "stapler":0, "hook":0, "other_consumables":0, "other_instrument":0,"total":0}



dic = {"other_instrument": 0, "other_consumables": 0, "hem_o_lok": 0, "clip": 0, "clip_applicator": 0,
       "suture_needle": 0, "specimen_bag": 0, "ultrasonic_knife": 0, "electrotome": 0, "stapler": 0, "gauze": 0,
       "forceps": 0, "attractor": 0, "needle_holder": 0, "line": 0, "scissors": 0, "electric_hook": 0,
       "total": 0}



def count_files(xml_file):
    dic = {'total':0}

def count(xml_file):

    xml_file = xmldom.parse(xml_file)
    # 获取xml文件中的元素
    tracks = xml_file.getElementsByTagName('track')
    # 获取xml中标签数量
    num_tracks = len(tracks)
    # 进度条展示
    with alive_bar(num_tracks) as bar:
        for track in tracks:
            label = track.getAttribute('label')
            if label not in dic:
                dic[label] = 1
            else:
                dic[label] += 1
            dic['total'] += 1
            bar()  # 显示进度下
    workbook = xw.Workbook('计数.xlsx')  # 创建工作簿
    worksheet1 = workbook.add_worksheet("sheet1")  # 创建子表
    worksheet1.activate()  # 激活表
    title = ['类别', '数量']  # 设置表头
    worksheet1.write_row("A1",title)
    row_num = 1
    for key, value in dic.items():
        row_num += 1
        row = 'A'+str(row_num)
        insert_data = [key,value]
        worksheet1.write_row(row, insert_data)
    workbook.close()
if __name__ == '__main__':

    dir_name = r"annotations/annotations.xml"
    count_files(dir_name)

    xml_name = "annotations.xml"
    count(xml_name)

