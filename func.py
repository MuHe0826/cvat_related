# 修改文件夹下所有视频的帧率
import os
import cv2

# 修改视频帧率为指定帧率，分辨率保持不变
def modify_video_frame_rate(videoPath, destFps):
    """
    :param videoPath: 视频路径
    :param destFps: 目标帧率
    """
    dir_name = os.path.dirname(videoPath)
    basename = os.path.basename(videoPath)
    video_name = basename[:basename.rfind('.')]
    video_name = video_name + "modify_fps_rate"
    resultVideoPath = f'{dir_name}/{video_name}.mp4'

    videoCapture = cv2.VideoCapture(videoPath)

    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    if fps != destFps:
        frameSize = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # 这里的VideoWriter_fourcc需要多测试，如果编码器不对则会提示报错，根据报错信息修改编码器即可
        videoWriter = cv2.VideoWriter(resultVideoPath, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), destFps, frameSize)
        i = 0
        while True:
            success, frame = videoCapture.read()
            if success:
                i += 1
                print('转换到视频{}的第{}帧'.format(video_name, i))
                videoWriter.write(frame)
            else:
                print('帧率转换结束')
                break
        # 关闭视频文件
        videoCapture.release()
        # 关闭所有 OpenCV 窗口
        cv2.destroyAllWindows()
        os.remove(videoPath)
        print("文件{}已删除".format(videoPath))


if __name__ == '__main__':
    source_video = "video"
    files = [file for file in os.listdir(source_video)]
    for file in files:
        modify_video_frame_rate('{}/{}'.format(source_video, file), 25)
