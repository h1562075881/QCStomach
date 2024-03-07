# cython: language_level=3
# -*- coding: utf-8 -*-
import sys
import traceback

import cv2

from QCStomach import QCStomach

class QCService(object):
    def __init__(self, videoPath, modelPath):
        # 视频设置
        self.videoPath = videoPath #播放视频文件路径
        self.modelPath = modelPath #胃质控模型路径
        self.originFrameWidth = 1920
        self.originFrameHeight = 1080
        
        #####初始化其他参数####
        self.isDebug = True
        self.isShowVideo = False
        self.currentFrameIndex = 0

        self.aiService = QCStomach(modelPath)

    def GetInput(self):
        videoCapture = cv2.VideoCapture(self.videoPath)
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        
        #opencv默认读摄像头的宽高为640*480 fps默认为0
        videoCapture.set(cv2.CAP_PROP_FRAME_WIDTH, self.originFrameWidth) 
        videoCapture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.originFrameHeight)
        videoCapture.set(cv2.CAP_PROP_FPS, 25)
        if not videoCapture.isOpened():
            print("视频打开异常，程序自动退出")
            return False, videoCapture
        return True, videoCapture

    def StartService(self):
        """
        主循环
        :return:
        """
        res, videoCapture = self.GetInput()
        if not res:
            sys.exit(1)

        try:
            while videoCapture.isOpened():
                capReturn, originFrame = videoCapture.read()
                if not capReturn:
                    print("视频帧获取异常")
                    break
 
                self.aiService.SetInput(originFrame)
                list_Result = self.aiService.GetResult()
                if len(list_Result) > 0:
                    for res in list_Result:
                        print(res)
                self._ShowVideo(originFrame)

                key = cv2.waitKey(1)
                if key == 27: #esc
                    break
        except KeyboardInterrupt: #按下Ctrl+C
            print("用户按压Ctrl+C,程序退出")
        except:
            traceback.print_exc()  
        finally:
            sys.exit(1)


    def _ShowVideo(self,originFrame):
        #注意：画的框并不精确，因为病变结果列表是把多张图预测的结果组合在一起返回的
        if self.isShowVideo:
            cv2.namedWindow('video', cv2.WND_PROP_FULLSCREEN)
            cv2.moveWindow('video', 0, 0)
            cv2.setWindowProperty('video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.imshow("video", originFrame)

