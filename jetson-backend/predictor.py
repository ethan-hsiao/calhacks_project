import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Predictor():
    def __init__(self, config_path, weights_path):
        self.config_path = config_path
        self.weights_path = weights_path
        
        with open(config_path) as config_buffer:    
            self.config = json.load(config_buffer)
        
        self.yolo = YOLO(backend             = self.config['model']['backend'],
                    input_size          = self.config['model']['input_size'], 
                    labels              = self.config['model']['labels'], 
                    max_box_per_image   = self.config['model']['max_box_per_image'],
                    anchors             = self.config['model']['anchors'])
        
        self.yolo.load_weights(self.weights_path)
    
    def predict(self, image):
        boxes = self.yolo.predict(image)
        image = draw_boxes(image, boxes, self.config['model']['labels'])
        return boxes, image