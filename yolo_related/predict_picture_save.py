# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# # Load a model
#
# model = YOLO('/home/zj/ultralytics/runs/segment/train5/weights/best.pt')  # load a custom model
#
# # Predict with the model
# results = model('/home/zj/ultralytics/datasets/test/images/frame_000272.png')  # predict on an image
# plt.show(results)
import cv2

from ultralytics import YOLO
import os

from ultralytics.utils.plotting import Annotator, colors
# Load a model
# Customize validation settings
def predict_picture(pt_path,source_img,target_img):
    if not os.path.exists(target_img):
        os.makedirs(target_img)
    model = YOLO(pt_path)
    names = model.model.names
    img_list = os.listdir(source_img)
    for img in img_list:
        img_path = os.path.join(source_img,img)
        im0 = cv2.imread(img_path)
        results = model.predict(im0, device='cuda')
        annotator = Annotator(im0, line_width=1)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            if results[0].masks.xy is not None:

                masks = results[0].masks.xy
            else:
                continue
            try:
                for mask, cls in zip(masks, clss):
                    annotator.seg_bbox(mask=mask,
                                       mask_color=colors(int(cls), True),
                                       det_label=names[int(cls)])
            except Exception as e:
                print("Exception")
        #print("Saved {}".format(os.path.join(target_img, img)))
        if cv2.imwrite(os.path.join(target_img,img),im0):
            print("Saved {}".format(os.path.join(target_img,img)))
if __name__ == "__main__":
    predict_picture(r'C:\Users\jozon\Desktop\pt\best_5_23.pt',r"C:\Users\jozon\Desktop\valid\test\images",r"C:\Users\jozon\Desktop\valid\test\show")