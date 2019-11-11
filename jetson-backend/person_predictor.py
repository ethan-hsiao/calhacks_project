import argparse
import tensorflow as tf
import cv2

from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame
from core.yolo_tiny import YOLOv3_tiny
from core.yolo import YOLOv3

class Predictor():

  def __init__(self, iou_threshold, confidence_threshold):
    self.class_names, self.n_classes = load_class_names()
    self.model = YOLOv3(n_classes=self.n_classes,
                     iou_threshold=iou_threshold,
                     confidence_threshold=confidence_threshold)
    self.inputs = tf.placeholder(tf.float32, [1, *self.model.input_size, 3])
    self.detections = self.model(self.inputs)
    self.saver = tf.train.Saver(tf.global_variables(scope=self.model.scope))
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    self.sess = tf.Session(config=config)
    self.saver.restore(self.sess, './weights/model.ckpt')
  
  def predict(self, frame):
    resized_frame = cv2.resize(frame, dsize=tuple((x) for x in self.model.input_size[::-1]), interpolation=cv2.INTER_NEAREST)
    result = self.sess.run(self.detections, feed_dict={self.inputs: [resized_frame]})
    #print(result['person'])
    # num = draw_boxes_frame(frame, frame.shape, result, self.class_names, self.model.input_size, 'person')
    return frame, result

if __name__ == "__main__":
  model = Predictor(0.5, 0.5)
  cam = cv2.VideoCapture(1)
  while True:
    ret, frame = cam.read()
    pred, res = model.predict(frame)
    cv2.imshow('frame', pred)
    cv2.waitKey(1)
