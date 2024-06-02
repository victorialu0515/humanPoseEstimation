import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import numpy as np

from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

import cv2
import json

from dataclasses import dataclass
from typing import TypedDict, List

@dataclass
class RockDetection:
  id: int
  bbox: List[float]
  colour: List[float]
  track: int

  def to_dict(self):
    return {"id":self.id, "bbox":self.bbox, "colour":self.colour, "track":self.track}

def getColour(image, result):
  colours = []
  imageArray = cv2.imread(image)
  for i in result:
    x, y, w, h = i

    img = imageArray[int(y-h/3):int(y+h/3), int(x-w/3):int(x+w/3)]
    average = img.mean(axis=0).mean(axis=0)

    pixels = np.float32(img.reshape(-1, 3))

    n_colors = 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
    _, counts = np.unique(labels, return_counts=True)

    dominant = palette[np.argmax(counts)]

    colours.append(dominant[::-1])
  return colours

def clustering(numClusters, colours):
    kmeans = KMeans(n_clusters=numClusters, random_state=None)
    kmeans.fit(colours)
    labels = kmeans.labels_
    return list(labels)

def drawColour(image, result, output):
    imageArray = cv2.imread(image)
    for i, rectangle in enumerate(result):
        x, y, w, h = rectangle['bbox']
        r, g, b = rectangle['colour']
        cv2.rectangle(imageArray, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), (int(b), int(g), int(r)), 3)
        cv2.putText(imageArray, str(rectangle['id']), (int(x), int(y)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=2)
        cv2.putText(imageArray, str(rectangle['track']), (int(x)-10, int(y)-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=2)
    cv2.imwrite(f"{output}.jpg", imageArray)


def detection(image_path, numCluster, output_path, model_path="/Users/victorialu/PycharmProjects/humanPoseEstimation/best.pt"):
    model  = YOLO(model_path)
    results = model(image_path)
    result = [x.tolist() for x in results[0].boxes.xywh]
    resultArray = np.array(result)
    colours = getColour(image_path, result)
    tracks = clustering(numCluster, colours)
    list_colours = [array.tolist() for array in colours]
    list_rocks: List[RockDetection] = [
        RockDetection(id=i, bbox=bbox, colour=colour, track=int(track)).to_dict()
        for i, (bbox, colour, track) in enumerate(zip(result, list_colours, tracks))
    ]

    result_dict = {
        "user_id": "Victoria",
        "run_id": 0,
        "track_chosen": 1,
        "rocks": list_rocks,
    }
    with open(f"{output_path}.json", "w") as f:
        json.dump(result_dict, f, indent=4)

    drawColour(image_path, result_dict["rocks"], output_path)


detection("/Users/victorialu/Downloads/IMG_2268_time_0.jpg", 4, "test3_without_person") #CHANGE

#loading file
# with open("detection_data.json") as f:
#   result_dict =json.load(f)
# drawColour(image_path, result_dict["rocks"], output_path)


