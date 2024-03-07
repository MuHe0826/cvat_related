**auxiliary_annotation.py 利用训练好的模型进行辅助标注，修改相应的annotations文件**

**Classification.py 对各类物体进行分类，存储在相应的文件夹下**

**counter.py 统计已经标记的各类物体的个数**

**createDataset.py 根据项目生成的annotations.xml文件来制作nnUNet所需要的数据集**

**func.py 是利用opencv对视频帧率进行转换，cvat帧率为25**

**getResult.py 根据annotations.xml文件生成标注结果**

**predicted.py 输入原始img和label生成标注后img**
