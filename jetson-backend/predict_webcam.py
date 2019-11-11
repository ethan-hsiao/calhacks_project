#! /usr/bin/env python

from predictor import Predictor
from person_predictor import Predictor as Person_Predictor
from graphs import Graph
import cv2
from matplotlib import pyplot as plt
from algorithms import *
from firebase_push import DBPusher
from core.utils import load_class_names, load_image, draw_boxes, draw_boxes_frame

def bboxes_to_points(boxes, image_w, image_h):
    return [(int((box.xmax+box.xmin)*image_w/2), int((box.ymax+box.ymin)*image_h/2)) for box in boxes]

if __name__=='__main__':
    cam = cv2.VideoCapture(0)
    pred = Predictor('config.json', 'model.h5')
    person_pred = Person_Predictor(0.5, 0.5)
    pusher = DBPusher()
    while True:
        # get frame
        ret, frame = cam.read()

        # predict humans and generate points
        boxes, image = pred.predict(frame)
        im, res = person_pred.predict(frame)
        num = draw_boxes_frame(frame, image.shape, res, person_pred.class_names, person_pred.model.input_size, 'person')
        image_h, image_w, _ = image.shape
        points = bboxes_to_points(boxes, image_w, image_h)
        print(len(res[0][num]))
        points_2 = [(int((box[0]+box[2])/2), int((box[1]+box[3])/2)) for box in res[0][num]]
        points.extend(points_2)
        if len(points) == 0: continue
        msts = Graph(coords=points).mst().prune_long_edges(1)

        clusters = []
        lines = []
        for mst in msts:
            r2 = poly_r_squared_of_points(mst.coords)
            print(r2)

            # Is a line or just a cluster
            if r2 > 0.4:
                line = []
                for edge in mst.edge_list:
                    image = cv2.line(image, mst.coords[edge[0]], mst.coords[edge[1]], (0, 255, 0), 3)
                    line.append(points[edge[0]])
                lines.append(line)
            else:
                if mst.get_max_length() > 8:
                    for cluster in k_means_classify(mst.coords):
                        center, radius = get_centroid_and_radius(cluster)
                        image = cv2.circle(image, center, radius, (255, 0, 0), 5)
                        clusters.append((center[0], center[1], radius, len(cluster)))
                else:
                    center, radius = get_centroid_and_radius(mst.coords)
                    image = cv2.circle(image, center, radius, (255, 0, 0), 5)
                    clusters.append((center[0], center[1], radius, len(mst.coords)))
        # print(boxes)
        # print(msts.edge_list)
        pusher.push(len(points), lines, clusters)
        print("Insights:")
        print('Num people total:', len(boxes))
        print('Clusters:',clusters)
        print('Lines', lines)
        print('--------------')

        cv2.imshow('frame', image)
        cv2.waitKey(1)
