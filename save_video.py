#!/usr/bin/env python

import numpy as np
import cv2
import time
import requests
import threading
from threading import Thread, Event, ThreadError

frames_needed = 400

class Cam():



  def __init__(self, url):
    
    print("camera starting")
    self.stream = requests.get(url, stream=True)
    self.thread_cancelled = False
    self.thread = Thread(target=self.run)
    print("camera initialised")
    # self.out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (2592, 1944))
    self.out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 2, (1920,1080))
    self.new_image = False
    
  def start(self):
    self.thread.start()
    print("camera stream started")
    
  def run(self):
    raw_data=b""
    count = 0
    while not self.thread_cancelled:
      try:
        d = self.stream.raw.read(512)
        raw_data+=d#self.stream.raw.read(1024)
        a = raw_data.find(b'\xff\xd8')
        b = raw_data.find(b'\xff\xd9')
        if a!=-1 and b!=-1:
          jpg = raw_data[a:b+2]
          raw_data= raw_data[b+2:]
          img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8),cv2.IMREAD_COLOR)
          cv2.imshow('cam', img)
          self.out.write(img)
          count += 1
          if cv2.waitKey(1) ==27:
            exit(0)
          print(count)
          if count >= frames_needed:
            print('frames grabbed, exiting')

            cv2.waitKey(0)
            self.out.release()
            break
      except ThreadError:
        self.thread_cancelled = True
        
        
  def is_running(self):
    return self.thread.isAlive()
      
    
  def shut_down(self):
    self.thread_cancelled = True
    self.out.release()

    #block while waiting for thread to terminate
    while self.thread.isAlive():
      time.sleep(1)
    return True

  
    
if __name__ == "__main__":
  url = 'http://192.168.1.120:8000/video_feed'
  url = 'http://192.168.1.120:8000/stream.mjpg'
  cam = Cam(url)
  cam.start()