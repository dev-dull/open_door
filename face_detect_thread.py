import os
import cv2

from threading import Thread
from datetime import datetime

class FaceDetectThread(Thread):
    def __init__(self, in_queue, out_queue, cascade_filenames, scale=1.05, min_neighbors=6, box_color=(0, 0x80, 0xFF)):
        Thread.__init__(self)
        self.box_color = box_color
        self.iq = in_queue
        self.oq = out_queue
        self.scale = scale
        self.min_neighbors = min_neighbors

        # TODO: put this into config.yaml
        cascade_filenames = ['haarcascade_frontalface_alt.xml',
                          'haarcascade_frontalface_alt2.xml',
                          'haarcascade_profileface.xml',
                          'haarcascade_frontalface_alt_tree.xml']
        # results in a lot of false-positives. Separated from the others so its easy to comment out.
        #cascade_filenames.append('haarcascade_frontalface_default.xml')
        self.cascades = self._get_cascades(cascade_filenames)

    def _get_cascades(self, face_cascade_files):
        cascade_dir = cv2.__file__.split(os.sep)
        cascade_dir[-1] = 'data'
        cascade_dir = os.sep.join(cascade_dir)

        face_cascades = {}
        for cascade_file in face_cascade_files:
            print(os.sep.join([cascade_dir, cascade_file]))
            face_cascades[cascade_file] = (cv2.CascadeClassifier(os.path.join(cascade_dir, cascade_file)))
        return face_cascades

    def run(self):
        while True:
            image = self.iq.get()
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for cascade_name,cascade in self.cascades.items():
                #found_faces = cascade.detectMultiScale(gray_image, scaleFactor=self.scale, minNeighbors=self.min_neighbors)
                found_faces = cascade.detectMultiScale(gray_image, minNeighbors=self.min_neighbors)
                if len(found_faces):
                    for (x,y,w,h) in found_faces:
                        cv2.rectangle(image, (x, y), (x+w, y+h), self.box_color, 2)
                        print(cascade_name, datetime.now(), x,y,w,h)
                    file_name = datetime.now().strftime('%Y%m%d%S%f.png')
                    cv2.imwrite('gray-' + file_name, gray_image)
                    self.oq.put(image)
                    break
            self.iq.task_done()
