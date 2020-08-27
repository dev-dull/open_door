import cv2

from time import time
from copy import copy
from queue import Queue
from threading import Thread
from cam_thread import CamThread
from flask import Flask, Response
from face_detect_thread import FaceDetectThread

class OpenDoor(Thread):
    def __init__(self, stream_queue):
        Thread.__init__(self)

        self.stream_queue = stream_queue

        self.start_time = time()
        self.cam_queue = Queue()
        self.find_queue = Queue()
        self.found_queue = Queue()

        self.camera = CamThread(self.cam_queue)
        self.facial = FaceDetectThread(self.find_queue, self.found_queue, ['TODO: read filenames from config'])

        self.camera.setDaemon(True)
        self.facial.setDaemon(True)

        self.camera.start()
        self.facial.start()

        self.fct = 0
        self.fps = 0
        self.pfct = 0
        self.pfps = 0

        self.score_average = 0x7fffffffffffffffe

    def image_diff(self, previous_image, new_image):
        diff_image = cv2.absdiff(previous_image, new_image)
        return sum(cv2.sumElems(diff_image)),diff_image

    def run(self):
        previous_image = self.cam_queue.get()
        while True:
            image = self.cam_queue.get()
            score,id = self.image_diff(previous_image, image)
            previous_image = image

            self.fct += 1
            current_time = time()
            elapsed_time = current_time - self.start_time

            #if self.find_queue.qsize() < 1:
            if score > self.score_average * 1.4:
                print('Put an image into the find queue')
                self.find_queue.put(image)
                self.pfct += 1
                self.pfps = ((self.pfct/elapsed_time)+self.pfps)/2
            else:
                self.score_average = (score + self.score_average)/2

            if self.found_queue.qsize() > 0:
                image = self.found_queue.get()
                self.found_queue.task_done()

            self.fps = ((self.fct/elapsed_time)+self.fps)/2
            if elapsed_time >= 1:
                self.fct = 0
                self.pfct = 0
                self.start_time = current_time

            cp_image = copy(image)
            #cp_image = copy(id)
            cp_image = cv2.putText(cp_image, 'FPS: %.3f' % self.fps,  (5,15), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1, cv2.LINE_AA)
            cp_image = cv2.putText(cp_image, 'PFPS: %.3f' % self.pfps, (5,30), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1, cv2.LINE_AA)
            self.stream_queue.put(cp_image)
            self.cam_queue.task_done()
            self.stream_queue.join()


if __name__ == "__main__":
    stream_queue = Queue()
    od = OpenDoor(stream_queue)
    od.setDaemon(True)
    od.start()
    app = Flask('open_door')

    def _process_frame():
        while True:
            image = stream_queue.get()
            stream_queue.task_done()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + cv2.imencode('.jpg', image)[1].tostring() + b'\r\n')

    @app.route('/')
    def mjpeg_stream():
        return Response(_process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

    app.run(host='0.0.0.0', port=4322)
