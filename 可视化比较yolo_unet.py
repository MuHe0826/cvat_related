import re
from pyecharts import options as opts
from pyecharts.charts import Bar
import json
from pyecharts import options as opts
from pyecharts.charts import Bar, Page
import math
from decimal import Decimal
import yaml
# 从 JSON 文件中读取数据
def draw_picture():
    page = Page(layout=Page.DraggablePageLayout)
    with open(r'C:\Users\jozon\Desktop\valid\yolo_summary_behind_two_cls.json', 'r') as f:
        data_yolo = json.load(f)
    with open(r'D:\nnUNet\DATASET\nnUNet_results\Dataset666_InstrumentSegmentation\nnUNetTrainer__nnUNetPlans__2d\fold_0\validation\summary.json', 'r') as f:
        data_unet = json.load(f)
    # 访问数据
    foreground_mean_yolo = data_yolo['foreground_mean']
    foreground_mean_unet = data_unet['foreground_mean']
    mean_yolo = data_yolo['mean']
    mean_unet= data_unet['mean']
    mean_yolo = dict(sorted({int(re.findall(r'\d+',key)[0]):value for key,value in mean_yolo.items()}.items()))
    #mean_unet = dict(sorted({int(re.findall(r'\d+', key)[0]): value for key, value in mean_unet.items()}.items()))
    mean_unet_t = dict(sorted({int(re.findall(r'\d+',key)[0]): value for key, value in mean_unet.items()}.items()))
    mean_unet.clear()
    for key,value in mean_unet_t.items():
        if key==1:
            mean_unet[2]= value
        else:
            mean_unet[1]=value
    mean_unet = dict(sorted(mean_unet.items()))
    foreground_mean_x = ["Dice","IoU"]

    foreground_mean_y_yolo = [Decimal(foreground_mean_yolo["Dice"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"),
                              Decimal(foreground_mean_yolo["IoU"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP")]
    foreground_mean_y_unet = [Decimal(foreground_mean_unet["Dice"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"),
                              Decimal(foreground_mean_unet["Dice"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP")]
    e = (
        Bar()
        .add_xaxis(
            foreground_mean_x
        )
        .add_yaxis("YOLO", foreground_mean_y_yolo)
        .add_yaxis("UNET", foreground_mean_y_unet)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="前景平均指标", subtitle="YoLo_Unet横向比较"),
        )

    )
    #page.add(e)
    x = list(mean_yolo.keys())
    y_yolo_dice = []
    y_unet_dice = []
    y_yolo_iou = []
    y_unet_iou = []
    for key, value in mean_yolo.items():
        y_yolo_dice.append(Decimal(value["Dice"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"))
        y_yolo_iou.append(Decimal(value["IoU"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"))
    for key, value in mean_unet.items():
        y_unet_dice.append(Decimal(value["Dice"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"))
        y_unet_iou.append(Decimal(value["IoU"]).quantize(Decimal('0.01'), rounding="ROUND_HALF_UP"))
    c = (
        Bar()
        .add_xaxis(
            x
        )
        .add_yaxis("YOLO", y_yolo_iou)
        .add_yaxis("UNET", y_unet_iou)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="IoU系数", subtitle="YoLo_Unet横向比较"),
        )

    )
    d = (
        Bar()
        .add_xaxis(
            x
        )
        .add_yaxis("YOLO", y_yolo_dice)
        .add_yaxis("UNET", y_unet_dice)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=-15)),
            title_opts=opts.TitleOpts(title="Dice系数", subtitle="YoLo_Unet横向比较"),
        )

    )
    page.add(c)
    page.add(d)

    page.render("YOLO_UNET两类都后处理横向比较.html")

def print_label():
    with open(r"C:\Users\jozon\Desktop\InstrumentSegment01.yaml", "r", encoding="utf-8") as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
    labels = {value: key + 1 for key, value in result["names"].items()}
    for k, v in labels.items():
        print("{} : {}".format(v,k))
if __name__ == "__main__":
    #print_label()
    draw_picture()