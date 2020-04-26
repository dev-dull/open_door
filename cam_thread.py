import cv2

from queue import Queue
from threading import Thread

class CamThread(Thread):
    #width = int(1920*.2)
    #height = int(1080*.2)
    #def __init__(self, queue, resolution=(384,288)):
    def __init__(self, queue, resolution=(640,480)):
    #def __init__(self, queue, resolution=(1024,768)):
        Thread.__init__(self)
        self.q = queue
        self.resolution = resolution

        capture_device = cv2.VideoCapture(-1)
        self.capture_device = self.set_resolution(capture_device, resolution[0], resolution[1])

    def set_resolution(self, capture_device, width, height):
        capture_device.set(3, width)
        capture_device.set(4, height)
        return capture_device

    def run(self):
        while True:
            ret, image = self.capture_device.read()
            if self.q.qsize() < 30:
                self.q.put(image)
            #self.q.join()
