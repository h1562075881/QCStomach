# cython: language_level=3
import numpy as np
import torch
import cv2
import queue

from Xception import BuildXception


class QCStomach(object):
    def __init__(self, modelPath):
        self.modelPath = modelPath
        self.inputQueue = queue.Queue(maxsize=3)
        # 参数设定
        self.inputImgSize = 256
        self.resultList = []

        # 加载模型
        self.model = BuildXception(29)
        self.model.load_state_dict(torch.load(self.modelPath))
        self.model = self.model.cuda().eval()

    def LetterboxResize(self, image, imgMaxSize):
        """keep aspect ratio resize
        image: input image
        imgMaxSize: a int number, the resized image's max size
        """
        originShape = image.shape[:2]  # current shape [height, width]
        # Scale ratio (new / old)
        r = min(imgMaxSize / originShape[0], imgMaxSize / originShape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(originShape[1] * r)), int(round(originShape[0] * r))
        dw, dh = imgMaxSize - new_unpad[0], imgMaxSize - new_unpad[1]  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if originShape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        return image, ratio, (dw, dh)

    def preprocessedPic(self,frame):
        # 数据预处理
        # 预处理,缩放当前帧，转换颜色空间到RGB，并进行空间变换
        resizedFrame, _, _ = self.LetterboxResize(frame, self.inputImgSize)
        resizedFrame = cv2.cvtColor(resizedFrame, cv2.COLOR_BGR2RGB)
        # 数据预处理
        resizedFrame = resizedFrame.astype(np.float32) / 255.
        resizedFrame = np.moveaxis(resizedFrame, -1, 0)
        return resizedFrame
        
    def SetInput(self, currentFrame):
        # 入队视频帧缓存队列，如果队列满了则进行一次预测，若没有满则继续入队
        if self.inputQueue.full():
            list_inputQueue = list(self.inputQueue.queue)
            # 预测一次后清空输入缓存队列
            with self.inputQueue.mutex:
                self.inputQueue.queue.clear()
            # 预测
            self._Predict(list_inputQueue)
            
        self.inputQueue.put(currentFrame)

    
    def _Predict(self,list_inputQueue):
        # 每次预测前都重置resultList
        self.resultList.clear()
        batchInput =[self.preprocessedPic(img.copy()) for img in list_inputQueue]
        
        batchInput = torch.from_numpy(np.stack(batchInput)).cuda()
        with torch.no_grad():
            predLogits = self.model(batchInput)
            predProbs = torch.softmax(predLogits, 1)
            predProbs = np.array(predProbs.cpu())  # 传到CPU上
            predLabels = np.argmax(predProbs, 1)

            for i in range(predLabels.shape[0]):
                tmpLabel = predLabels[i]
                tmpProbList = predProbs[i]
                tmpProb = tmpProbList[tmpLabel]
                self.resultList.append([tmpLabel, tmpProb])

    def GetResult(self):
        result = self.resultList.copy()
        self.resultList.clear()
        return result