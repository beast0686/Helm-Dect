import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import cv2

# Display the input image
img0 = r"D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\detect-person-on-motorbike-or-scooter\dataset\obj\115.jpg"
_ = plt.figure(figsize=(15, 15))
_ = plt.axis('off')
_ = plt.imshow(mpimg.imread(img0))

# Directory containing images and corresponding box data
directory = r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\detect-person-on-motorbike-or-scooter\dataset\obj'

imagepath = []
imagefile = []
boxset = []
boxfile = []

# Gather image files
for im in os.listdir(directory):
    if im.endswith('.jpg'):
        path = os.path.join(directory, im)
        imagepath.append(path)
        imagefile.append(im)

# Gather corresponding box files
for im in imagefile:
    if im.endswith('.jpg'):
        bx = im[:-4] + '.txt'
        path = os.path.join(directory, bx)
        if os.path.isfile(path):
            bxdata = np.loadtxt(path)
            boxset.append(bxdata)
            boxfile.append(bx)

print(imagefile[:5])
print(boxfile[:5])
print(len(boxset))
print(len(imagepath))

# Find the index of the image of interest
num0 = 0
for i in range(len(imagepath)):
    if imagepath[i] == img0:
        num0 = i
        print(i)

# Paths to YOLO configuration and weight files
weights0_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\detect-person-on-motorbike-or-scooter\yolov3-obj_final.weights'
configuration0_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\detect-person-on-motorbike-or-scooter\yolov3_pb.cfg'

probability_minimum = 0.5
threshold = 0.3

# Load YOLO network for person detection
network0 = cv2.dnn.readNetFromDarknet(configuration0_path, weights0_path)
layers_names0_all = network0.getLayerNames()
layers_names0_output = [layers_names0_all[i - 1] for i in network0.getUnconnectedOutLayers()]
labels0 = open(r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\detect-person-on-motorbike-or-scooter\coco.names').read().strip().split('\n')
print(labels0)

# Paths to YOLO configuration and weight files for helmet detection
weights1_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\helmet-detection-yolov3\yolov3-helmet.weights'
configuration1_path = r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\helmet-detection-yolov3\yolov3-helmet.cfg'

# Load YOLO network for helmet detection
network1 = cv2.dnn.readNetFromDarknet(configuration1_path, weights1_path)
layers_names1_all = network1.getLayerNames()
layers_names1_output = [layers_names1_all[i - 1] for i in network1.getUnconnectedOutLayers()]
labels1 = open(r'D:\BNMIT\Semester 4\Python\Project\Helmet Detection\datasets\helmet-detection-yolov3\helmet.names').read().strip().split('\n')
print(labels1)

# Prepare the input image
image_input = cv2.imread(imagepath[num0])
blob = cv2.dnn.blobFromImage(image_input, 1 / 255.0, (416, 416), swapRB=True, crop=False)
blob_to_show = blob[0, :, :, :].transpose(1, 2, 0)
network0.setInput(blob)
network1.setInput(blob)
output_from_network0 = network0.forward(layers_names0_output)
output_from_network1 = network1.forward(layers_names1_output)
np.random.seed(42)
colours0 = np.random.randint(0, 255, size=(len(labels0), 3), dtype='uint8')
colours1 = np.random.randint(0, 255, size=(len(labels1), 3), dtype='uint8')

print(colours0)
print(colours1)

bounding_boxes0 = []
confidences0 = []
class_numbers0 = []

bounding_boxes1 = []
confidences1 = []
class_numbers1 = []

h, w = image_input.shape[:2]

# Process detection results for person detection
for result in output_from_network0:
    for detection in result:
        scores = detection[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        if confidence_current > probability_minimum:
            box_current = detection[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current.astype('int')
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes0.append([x_min, y_min, int(box_width), int(box_height)])
            confidences0.append(float(confidence_current))
            class_numbers0.append(class_current)

# Process detection results for helmet detection
for result in output_from_network1:
    for detection in result:
        scores = detection[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]
        if confidence_current > probability_minimum:
            box_current = detection[0:4] * np.array([w, h, w, h])
            x_center, y_center, box_width, box_height = box_current.astype('int')
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes1.append([x_min, y_min, int(box_width), int(box_height)])
            confidences1.append(float(confidence_current))
            class_numbers1.append(class_current)

# Non-Maximum Suppression for person detection
results0 = cv2.dnn.NMSBoxes(bounding_boxes0, confidences0, probability_minimum, threshold)

if len(results0) > 0:
    for i in results0.flatten():
        x_min, y_min = bounding_boxes0[i][0], bounding_boxes0[i][1]
        box_width, box_height = bounding_boxes0[i][2], bounding_boxes0[i][3]
        colour_box_current = [int(j) for j in colours0[class_numbers0[i]]]
        cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 5)
        text_box_current0 = '{}: {:.4f}'.format(labels0[int(class_numbers0[i])], confidences0[i])
        cv2.putText(image_input, text_box_current0, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    colour_box_current, 5)

# Non-Maximum Suppression for helmet detection
results1 = cv2.dnn.NMSBoxes(bounding_boxes1, confidences1, probability_minimum, threshold)

if len(results1) > 0:
    for i in results1.flatten():
        x_min, y_min = bounding_boxes1[i][0], bounding_boxes1[i][1]
        box_width, box_height = bounding_boxes1[i][2], bounding_boxes1[i][3]
        colour_box_current = [int(j) for j in colours1[class_numbers1[i]]]
        cv2.rectangle(image_input, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 5)
        text_box_current1 = '{}: {:.4f}'.format(labels1[int(class_numbers1[i])], confidences1[i])
        cv2.putText(image_input, text_box_current1, (x_min, y_min - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    colour_box_current, 5)

# Display the output image
plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
plt.show()
