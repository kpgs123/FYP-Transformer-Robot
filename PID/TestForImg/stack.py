import cv2
import threading
import time

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.lock = threading.Lock()
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # grab frames as soon as they are available
    def _reader(self):
        while True:
            with self.lock:
                ret = self.cap.grab()
            if not ret:
                break

    # retrieve latest frame
    def read(self):
        with self.lock:
            _, frame = self.cap.retrieve()
        return frame
    
cap = VideoCapture(0)
while True:
  time.sleep(1)   # simulate time between events
  frame = cap.read()
  cv2.imshow("frame", frame)
  if chr(cv2.waitKey(1)&255) == 'q':
    break