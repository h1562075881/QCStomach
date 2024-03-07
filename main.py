# -*- coding: utf-8 -*-

from MainService import QCService

def main():
    video_path = r"C:\20201228_112623.mp4"
    model_path = r"C:\QCStomach.pth"
    s = QCService(video_path,model_path)
    s.StartService()
    
if __name__ == "__main__":
    main()
