import cv2
import json
from imread_from_url import imread_from_url

from HRNET import HRNET, ModelType
import copy
import numpy as np

with open("test3_without_person.json") as f: #CHANGE
  result_dict =json.load(f)
result = [rock['bbox'] for rock in result_dict["rocks"]]
# Initialize inference model
model_path = "models/hrnet_coco_w48_384x288.onnx"
# model_path = "models/litehrnet_30_coco_Nx384x288.onnx"
model_type = ModelType.COCO
hrnet = HRNET(model_path, model_type, conf_thres=0.6)

# Read image
# img_url = "https://upload.wikimedia.org/wikipedia/commons/8/8e/17_Years_of_Sekar_Jepun_2014-11-01_72.jpg"
#
# img = imread_from_url(img_url)

# Read image from disk
# img_path = "/Users/victorialu/Downloads/with person.jpg"
img_path = "/Users/victorialu/Downloads/IMG_2268_time_5.jpg" #CHANGE
img = cv2.imread(img_path)

# Perform the inference in the image
total_heatmap, peaks = hrnet(img)


def draw(output_img, result):
  for i, rectangle in enumerate(result):
    x = rectangle[0]
    y = rectangle[1]
    w = rectangle[2]
    h = rectangle[3]
    cv2.rectangle(output_img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 3)


# Draw Model Output
output_img = hrnet.draw_pose(img)
cv2.circle(output_img, (int(hrnet.poses[16][0]), int(hrnet.poses[16][1])), radius=10, color=(255, 0, 0), thickness=5)
draw(output_img, result)
cv2.namedWindow("Model Output", cv2.WINDOW_NORMAL)
cv2.imshow("Model Output", output_img)
cv2.imwrite("doc/img/output.jpg", output_img)
cv2.waitKey(0)

person_size = 700 #pixels
landmarks = copy.deepcopy(hrnet.poses)
# landmarks[landmark < -1E6] = np.mean()

leftHand = (int(hrnet.poses[9][0]), int(hrnet.poses[9][1]))
rightHand = (int(hrnet.poses[10][0]), int(hrnet.poses[10][1]))
leftFoot = (int(hrnet.poses[15][0]), int(hrnet.poses[15][1]))
rightFoot = (int(hrnet.poses[16][0]), int(hrnet.poses[16][1]))

error = 0.05 * person_size
def inBox(point, result):
  iList = []
  for i, box in enumerate(result):
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]
    p1, p2 = point
    if (p1 > x-w/2-error and p1 < x+w/2+error and p2 < y+h/2+error and p2 > y-h/2-error):
      iList.append(i)
  return iList

print(inBox(leftFoot, result))
print(inBox(rightFoot, result))
print(inBox(leftHand, result))
print(inBox(rightHand, result))
#lefthand:7, leftFoot:1
#test1: good
#test2: hrnet didn't detect right arm


# def detect(imagePath, personHeight, model="models/hrnet_coco_w48_384x288.onnx"):
#   img = cv2.imread(imagePath)
