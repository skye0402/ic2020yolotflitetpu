# Multi-object tracking for traffic using tflite instead of Detection Engine

# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import FileVideoStream
from PIL import Image
import argparse
import imutils
from time import time
from uuid import uuid4
import numpy as np
import cv2
import tflite_runtime.interpreter as tflite
import detect
import platform

EDGETPU_SHARED_LIB = {
  'Linux': 'libedgetpu.so.1',
  'Darwin': 'libedgetpu.1.dylib',
  'Windows': 'edgetpu.dll'
}[platform.system()]

def make_interpreter(tpu, model_file):
    if tpu:
        model_file, *device = model_file.split('@')
        return tflite.Interpreter(
            model_path=model_file,
            experimental_delegates=[
                tflite.load_delegate(EDGETPU_SHARED_LIB,
                                    {'device': device[0]} if device else {})
        ])
    else:
        return tflite.Interpreter(model_file)

def resizeAndPadImage(image,size):
    old_size = image.shape[:2] # old_size is in (height, width) format
    ratio = float(size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = size - new_size[1]
    delta_h = size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def get_anchors(file):
    #Load the anchors into an np array
    return np.loadtxt(file, delimiter=",")

def get_labels(file):
    # Load the classes text file
    with open(file) as f:
        content = f.read().splitlines()
    return content   

def featuresToBoxes(outputs, anchors, n_classes, net_input_shape, 
        img_orig_shape, threshold):
    grid_shape = outputs.shape[1:3]
    n_anchors = len(anchors)

    # Numpy screwaround to get the boxes in reasonable amount of time
    grid_y = np.tile(np.arange(grid_shape[0]).reshape(-1, 1), grid_shape[0]).reshape(1, grid_shape[0], grid_shape[0], 1).astype(np.float32)
    grid_x = grid_y.copy().T.reshape(1, grid_shape[0], grid_shape[1], 1).astype(np.float32)
    outputs = outputs.reshape(1, grid_shape[0], grid_shape[1], n_anchors, -1)
    _anchors = anchors.reshape(1, 1, 3, 2).astype(np.float32)

    # Get box parameters from network output and apply transformations
    bx = (sigmoid(outputs[..., 0]) + grid_x) / grid_shape[0] 
    by = (sigmoid(outputs[..., 1]) + grid_y) / grid_shape[1]
    # Should these be inverted?
    bw = np.multiply(_anchors[..., 0] / net_input_shape[1], np.exp(outputs[..., 2]))
    bh = np.multiply(_anchors[..., 1] / net_input_shape[2], np.exp(outputs[..., 3]))
    
    # Get the scores 
    scores = sigmoid(np.expand_dims(outputs[..., 4], -1)) * \
             sigmoid(outputs[..., 5:])
    scores = scores.reshape(-1, n_classes)

    # Reshape boxes and scale back to original image size
    ratio = net_input_shape[2] / img_orig_shape[1]
    letterboxed_height = ratio * img_orig_shape[0] 
    scale = net_input_shape[1] / letterboxed_height
    offset = (net_input_shape[1] - letterboxed_height) / 2 / net_input_shape[1]
    bx = bx.flatten()
    by = (by.flatten() - offset) * scale
    bw = bw.flatten()
    bh = bh.flatten() * scale
    half_bw = bw / 2.
    half_bh = bh / 2.

    tl_x = np.multiply(bx - half_bw, img_orig_shape[1])
    tl_y = np.multiply(by - half_bh, img_orig_shape[0]) 
    br_x = np.multiply(bx + half_bw, img_orig_shape[1])
    br_y = np.multiply(by + half_bh, img_orig_shape[0])

    # Get indices of boxes with score higher than threshold
    indices = np.argwhere(scores >= threshold)
    selected_boxes = []
    selected_scores = []
    for i in indices:
        i = tuple(i)
        selected_boxes.append( ((tl_x[i[0]], tl_y[i[0]]), (br_x[i[0]], br_y[i[0]])) )
        selected_scores.append(scores[i])

    selected_boxes = np.array(selected_boxes)
    selected_scores = np.array(selected_scores)
    selected_classes = indices[:, 1]

    return selected_boxes, selected_scores, selected_classes

def get_interpreter_details(interpreter):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]["shape"]

    return input_details, output_details, input_shape

# this class holds the detected objects and tracks them
class TrafficObject:
    # constructor method
    def __init__(self, tTracker, bBox, classLabel, creditLimit):
        self.id = str(uuid4()) # assign a unique ID
        self.tracker = tTracker
        self.box = bBox
        self.detectionCredit = creditLimit # start with the limit
        self.creditLimit = creditLimit
        self.labels = [classLabel]
    
    # Calculates the intersection over union (class method)
    def __calcIoU(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    # Create tracker object (class method)
    def createTracker(trackerType):
        if trackerType == 'BOOSTING':
            newTracker = cv2.TrackerBoosting_create()
        if trackerType == 'MIL':
            newTracker = cv2.TrackerMIL_create()
        if trackerType == 'KCF':
            newTracker = cv2.TrackerKCF_create()
        if trackerType == 'TLD':
            newTracker = cv2.TrackerTLD_create()
        if trackerType == 'MEDIANFLOW':
            newTracker = cv2.TrackerMedianFlow_create()
        if trackerType == 'GOTURN':
            newTracker = cv2.TrackerGOTURN_create()
        if trackerType == 'MOSSE':
            newTracker = cv2.TrackerMOSSE_create()
        if trackerType == "CSRT":
            newTracker = cv2.TrackerCSRT_create()
        return newTracker

    # Get IoU of tracker box vs detection
    def getIoU(self, boxInput):
        
        return self.__calcIoU(boxInput, self.__getBBox())

    # Get object ID
    def getId(self):
        return self.id

    # Add detection counter
    def addCount(self, tracker, image, bBox, credit):
        if self.detectionCredit < (maxCredit + detectionCredit):
            self.detectionCredit += credit
        self.tracker = tracker
        self.tracker.init(image, bBox) #Note! We initialize the tracker to adjust the bounding box size

    # Add label
    def addLabel(self, classLabel):
        self.labels.append(classLabel)

    # Get bounding box of tracker
    def __getBBox(self):
        return self.box

    # Get credit factor of tracker (must be greater than zero)
    def getCredit(self):
        creditFactor = int(100 * self.detectionCredit / self.creditLimit)
        if creditFactor < 0:
            creditFactor = 0
        return creditFactor

    # Get bounding box absolute coordinates
    def getBBoxCoord(self):
        return (self.box[0], self.box[1]), (self.box[2], self.box[3])

    # Update tracker with new frame
    def updateTracker(self, image):
        ok, bbox = self.tracker.update(image)
        self.detectionCredit -= 1 # We remove the credit from the last detection
        if ok:
            self.box = (int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3]))
            return True
        else:
            return False #tracker lost track


# Tracker related code
trafficDict = {} # Contains the list of objects found
ioUThreshold = 0.4 # Intersection over Union threshold value
staleObject = 20 # Amount of frames without detection (leads to object deletion)
detectionCredit = 2 # points awarded for successful detection of object to tracker
maxCredit = 30 # Maximum credit that be achieved
trackerType = "MEDIANFLOW" # Tracker to be used
# End of Tracker related code    

writeVideo = False # output video to file


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", required=True, help="path to labels file")
ap.add_argument("-a", "--anchors", required=True, help="path to anchors for Yolo model")
ap.add_argument("-c", "--confidence", type=float, default=0.65, help="minimum probability to filter weak detections")
ap.add_argument("-v", "--video", required=True, help="filename of video for detection")
ap.add_argument("-t", "--tpu", action='store_true', help="TPU is present/ should be used")
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
anchors = get_anchors(args["anchors"])
labels = get_labels(args["labels"])
n_labels = len(labels)
# Generate random colors for each detection
colors = np.random.uniform(30, 255, size=(n_labels, 3))

# # loop over the class labels file
# for row in open(args["labels"]):
# 	# unpack the row and update the labels dictionary
# 	(classID, label) = row.strip().split(maxsplit=1)
# 	labels[int(classID)] = label.strip()

# load the detection model
print("[INFO] loading model...")
interpreter = make_interpreter(args["tpu"], args["model"])
input_details, output_details, net_input_shape = get_interpreter_details(interpreter)

interpreter.allocate_tensors()

# Get required input size of frame
imgSize = input_details[0]['shape'].max()

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")

# Load the video
vs = FileVideoStream(args["video"]).start()

# Optional: write video out
if writeVideo: 
    fourcc = cv2.VideoWriter_fourcc(*'MP42') #(*'MP42')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280,720))

# loop over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of x pixels
    frame = vs.read()
    
    start = time()
    fpstimer = cv2.getTickCount()

    frame = resizeAndPadImage(frame, imgSize)
    conv_time = time() - start #Time to convert the frame
    orig = frame.copy()
    
    # prepare the frame for object detection by converting (1) it
	# from BGR to RGB channel ordering and then (2) from a NumPy
	# array to PIL image format
    # start = time()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # frame = Image.fromarray(frame)
    # rgb_time = time() - start

	# make predictions on the input frame
    start = time()

    interpreter.set_tensor(input_details[0]['index'], [frame])
    interpreter.invoke()
    out0 = interpreter.get_tensor(output_details[0]['index'])
    out1 = interpreter.get_tensor(output_details[1]['index'])

    # If this is a quantized model, dequantize the outputs
    if out0.dtype.name == "uint8":#args.quant:
        # Dequantize output
        o1_scale, o1_zero = output_details[0]['quantization']
        out0 = (out0.astype(np.float32) - o1_zero) * o1_scale
        o2_scale, o2_zero = output_details[1]['quantization']
        out1 = (out1.astype(np.float32) - o2_zero) * o2_scale
    # results = model.detect_with_image(frame, threshold=args["confidence"],keep_aspect_ratio=True, relative_coord=False)
    inf_time = time() - start

    # Get boxes from outputs of network
    _boxes1, _scores1, _classes1 = featuresToBoxes(out0, anchors[[3, 4, 5]], 
            n_labels, net_input_shape, (imgSize, imgSize), 0.3)
    _boxes2, _scores2, _classes2 = featuresToBoxes(out1, anchors[[1, 2, 3]], 
            n_labels, net_input_shape, (imgSize, imgSize), 0.3)

    # This is needed to be able to append nicely when the output layers don't
    # return any boxes
    if _boxes1.shape[0] == 0:
        _boxes1 = np.empty([0, 2, 2])
        _scores1 = np.empty([0,])
        _classes1 = np.empty([0,])
    if _boxes2.shape[0] == 0:
        _boxes2 = np.empty([0, 2, 2])
        _scores2 = np.empty([0,])
        _classes2 = np.empty([0,])

    boxes = np.append(_boxes1, _boxes2, axis=0)
    scores = np.append(_scores1, _scores2, axis=0)
    classes = np.append(_classes1, _classes2, axis=0)

    # Update active trackers
    for objectID in list(trafficDict):
        ok = trafficDict[objectID].updateTracker(orig)
        if not ok:
            del trafficDict[objectID] #remove the tracker

    # loop over the results
    for r in boxes:
        # extract the bounding box and box and predicted class label
        box = r.flatten().astype("int")
        (startX, startY, endX, endY) = box
        label = labels[classes[0].astype("int")]
   
        # Tracking handling starts here
        tBox = (startX, startY, endX-startX, endY-startY) # Detected box in tracker format
        objectFound = False
        for objectID in list(trafficDict):
            if trafficDict[objectID].getIoU(box) > ioUThreshold:               
                trafficDict[objectID].addLabel(label)
                trafficDict[objectID].addCount(TrafficObject.createTracker(trackerType), orig, tBox, detectionCredit)
                objectFound = True

        if objectFound == False: #No matching tracker, let's add a new tracker
            tObject = TrafficObject(TrafficObject.createTracker(trackerType), box, label, staleObject)
            tObject.tracker.init(orig, tBox)
            trafficDict[tObject.getId] = tObject

    for objectID in list(trafficDict):
        if trafficDict[objectID].getCredit() == 0: #Remove stale objects
            del trafficDict[objectID]
        else:
            startXY, endXY = trafficDict[objectID].getBBoxCoord()
            cv2.rectangle(orig, startXY, endXY, (0, 255, 0), 1)
            y = startXY[1] - 15 if startXY[1] - 15 > 15 else startXY[1] + 15
            text = "ID {}: {}% credit".format(trafficDict[objectID].getId()[-4:] , trafficDict[objectID].getCredit())
            cv2.putText(orig, text, (startXY[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # Tracker handling ends here


        # print time required
        # print(f"Image resize: |{conv_time*1000}|ms. RGB conv.: |{rgb_time*1000}|ms. Inference: |{inf_time*1000}|ms.")
    
    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - fpstimer)
    text = "fps: {:.0f}".format(fps)
    cv2.putText(orig, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # show the output frame and wait for a key press
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    
    # Optional: write video
    if writeVideo: out.write(orig)

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
if writeVideo: out.release()
cv2.destroyAllWindows()
vs.stop()
