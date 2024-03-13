import os
import cv2
import numpy as np
from tqdm import tqdm


def layer_overlay(source_images, source_labels, output_dir):
    # 将一层透明标签覆盖到目标图片的物体上
    os.makedirs(output_dir, exist_ok=True)

    files = [file for file in os.listdir(source_images) if file.endswith(".png")]

    for file in files:
        print(file)
        image_path = os.path.join(source_images, file)
        label_path = os.path.join(source_labels, file)
        image = cv2.imread(image_path)

        # 根据预测的labels进行标注
        gray_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        rgb_label = cv2.cvtColor(gray_label, cv2.COLOR_GRAY2RGB)
        # 根据真正的labels进行标注
        # rgb_label = cv2.imread(label_path)

        r = np.zeros_like(rgb_label[:, :, 2])
        g = np.zeros_like(rgb_label[:, :, 1])
        b = np.zeros_like(rgb_label[:, :, 0])
        # 标签为1的是consumables,标为黄色
        r[rgb_label[:, :, 2] == 1] = 255
        g[rgb_label[:, :, 1] == 1] = 255
        b[rgb_label[:, :, 0] == 1] = 0
        # 标签为2的是instrument，标为蓝色
        r[rgb_label[:, :, 2] == 2] = 0
        g[rgb_label[:, :, 1] == 2] = 0
        b[rgb_label[:, :, 0] == 2] = 255
        # cv2读取的rgb图像按照b,g,r的顺序保存
        annotated_image = cv2.merge([b, g, r])

        alpha = 0.8
        meta = 1 - alpha
        gamma = 0
        image = cv2.addWeighted(image, alpha, annotated_image, meta, gamma)

        cv2.imwrite("{}/{}".format(output_dir, file), image)


def draw_outline(source_images, source_labels, output_dir):
    # 画出目标图片的物体的轮廓
    os.makedirs(output_dir, exist_ok=True)

    files = [file for file in os.listdir(source_images) if file.endswith(".png")]

    for file in files:
        print(file)
        image_path = os.path.join(source_images, file)
        label_path = os.path.join(source_labels, file)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

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


if __name__ == "__main__":
    source_images = "dataset/test/images"
    source_true_labels = "dataset/test/labels"
    source_pred_labels = "dataset/test_pred"
    output_dir = "pred"

    # layer_overlay(source_images, source_labels, output_dir)
    # draw_outline(source_images, source_labels, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    files = [file for file in os.listdir(source_images) if file.endswith(".png")]

    for file in tqdm(files, desc="Processing", unit="files"):
        image_path = os.path.join(source_images, file)
        true_label_path = os.path.join(source_true_labels, file)
        true_label = cv2.imread(true_label_path, cv2.IMREAD_GRAYSCALE)
        pred_label_path = os.path.join(source_pred_labels, file)
        pred_label = cv2.imread(pred_label_path, cv2.IMREAD_GRAYSCALE)

        image1 = cv2.imread(image_path)
        label1 = np.zeros_like(true_label)
        label1[true_label == 1] = 255
        contours, hierarchy = cv2.findContours(label1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image1, contours, -1, (0, 255, 0), thickness=2)
        cv2.imwrite("{}/{}".format(output_dir, file), image1)
        label2 = np.zeros_like(true_label)
        label2[true_label == 2] = 255
        contours, hierarchy = cv2.findContours(label2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image1, contours, -1, (255, 0, 0), thickness=2)

        image2 = cv2.imread(image_path)
        label1 = np.zeros_like(pred_label)
        label1[pred_label == 1] = 255
        contours, hierarchy = cv2.findContours(label1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image2, contours, -1, (0, 255, 0), thickness=2)
        cv2.imwrite("{}/{}".format(output_dir, file), image2)
        label2 = np.zeros_like(pred_label)
        label2[pred_label == 2] = 255
        contours, hierarchy = cv2.findContours(label2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image2, contours, -1, (255, 0, 0), thickness=2)

        # 将两张图片拼接在一起
        merged_image = cv2.hconcat([image1, image2])

        # 保存拼接后的图片
        cv2.imwrite("{}/{}".format(output_dir, file), merged_image)

