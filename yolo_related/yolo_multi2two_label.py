#将多分类标签改为二分类标签，yolo的输入为txt，读txt并将其中的标签内容分为0，1
import os
import random
import shutil


def change_label(txt_path):
    for txt_file in os.listdir(txt_path):
        if txt_file.endswith('.txt'):
            txt_file_path = os.path.join(txt_path, txt_file)
            with open(txt_file_path, 'r') as f:
                contents = f.readlines()
            with open(txt_file_path, "w") as f:
                for i, content in enumerate(contents):
                    parts = content.split()
                    if int(parts[0]) in [1, 2, 3, 5, 6, 10, 14, ]:  # 器械
                        parts[0] = '1'
                    else:
                        parts[0] = '0'
                    contents[i] = ' '.join(parts)
                    f.writelines(contents[i] + '\n')


def split_data(dataset_name):
    test_ratio = 0.1
    val_ratio = 0.1
    source_images = "{}/images".format(dataset_name)
    source_labels = "{}/labels".format(dataset_name)
    output_train_dir = "{}/train".format(dataset_name)
    output_val_dir = "{}/valid".format(dataset_name)
    output_test_dir = "{}/test".format(dataset_name)
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_val_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)
    # 创建训练集和测试集目录
    os.makedirs(output_train_dir + "/images", exist_ok=True)
    os.makedirs(output_train_dir + "/labels", exist_ok=True)
    os.makedirs(output_val_dir + "/images", exist_ok=True)
    os.makedirs(output_val_dir + "/labels", exist_ok=True)
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


if __name__ == "__main__":
    change_label("/home/zj/dataset/InstrumentSegment01_6_14_two_label/labels")
    split_data("/home/zj/dataset/InstrumentSegment01_6_14_two_label")