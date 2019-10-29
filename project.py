# pedestrian-detection source: https://github.com/djmv/MobilNet_SSD_opencv

# Import the neccesary libraries
import numpy as np
import argparse
import cv2

# Set path to video file
video = '01.avi'

# Set path to model
prototxt = "MobileNetSSD_deploy.prototxt"
# Set path to model pertained weights
weights = "MobileNetSSD_deploy.caffemodel"
# Set prediction confidence threshold
thr = 0.2

# Load the Caffe model
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

# Initial required variables
baseFrame = None
BasexLeftBottom = 0
BasexRightTop = 0
BaseyRightTop = 0
BaseyLeftBottom = 0

# Read the video
cap = cv2.VideoCapture(video)

# Capture frame-by-frame
ret, frame = cap.read()

# Set first human included frame as baseFrame to check motion
if baseFrame is None:
    # Apply a heavy blur in BaseFrame
    blur = cv2.GaussianBlur(frame, (43, 43), 0)
    baseFrame = blur


ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()
ret, frame = cap.read()



frame_resized = cv2.resize(frame, (300, 300))  # resize frame for prediction
# MobileNet requires fixed dimensions for input image(s)
# so we have to ensure that it is resized to 300x300 pixels.
# set a scale factor to image because network the objects has different size.
# We perform a mean subtraction (127.5, 127.5, 127.5) to normalize the input;
# after executing this command our "blob" now has the shape:
# (1, 3, 300, 300)
blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
# Set to network the input blob
net.setInput(blob)
# Get prediction of network
detections = net.forward()
# Size of frame resize (300x300)
cols = frame_resized.shape[1]
rows = frame_resized.shape[0]

# Set a flag to check if a human object detected in frame
found = False

# For each detection
for i in range(detections.shape[2]):
    # For get the class and location of object detected,
    # There is a fix index for class, location and confidence
    # value in detections array .
    confidence = detections[0, 0, i, 2]  # Confidence of prediction
    class_id = int(detections[0, 0, i, 1])  # Class label
    if class_id == 15 and confidence > thr:  # Filter prediction by thresholding confidence and human class label
        found = True
        # Object location
        xLeftBottom = int(detections[0, 0, i, 3] * cols)
        yLeftBottom = int(detections[0, 0, i, 4] * rows)
        xRightTop = int(detections[0, 0, i, 5] * cols)
        yRightTop = int(detections[0, 0, i, 6] * rows)
        # Factor for scale to original size of frame
        heightFactor = frame.shape[0] / 300.0
        widthFactor = frame.shape[1] / 300.0
        # Scale object detection to frame
        xLeftBottom = int(widthFactor * xLeftBottom)
        yLeftBottom = int(heightFactor * yLeftBottom)
        xRightTop = int(widthFactor * xRightTop)
        yRightTop = int(heightFactor * yRightTop)



# Create a binary mask by detected object bounding box
mask = np.full((frame.shape[0], frame.shape[1]), 0, dtype=np.uint8)
cv2.rectangle(mask, (xLeftBottom, yLeftBottom - 5), (xRightTop, yRightTop),
              255, -1)


# Calculates an absolute difference value BaseFrame and frame pixels
frameDelta = cv2.absdiff(frame, baseFrame)

# Apply created detection mask on frameDelta
frameDelta = cv2.bitwise_and(frameDelta, frameDelta, mask=mask)

# Convert frameData to greyscale
frameDelta = cv2.cvtColor(frameDelta, cv2.COLOR_RGB2GRAY)
# Apply Gaussian Blur on frameDelta
frameDelta = cv2.GaussianBlur(frameDelta, (5, 5), 0)


# Apply OTSU threshold to segment object in frameDelta
thresh1 = cv2.threshold(frameDelta, 20, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Apply Erosion Morphological transformation to remove small noises
thresh = cv2.morphologyEx(thresh1, cv2.MORPH_ERODE, np.ones((5, 3), np.uint8))



# Apply Dilation Morphological transformation to remove Erosion effect on main segment
thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, np.ones((3, 1), np.uint8))



# Apply Erosion Morphological transformation to remove fractious between parts
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((19, 9), np.uint8), iterations=2)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=5)
# Apply mask again to remove outbounded parts
thresh = cv2.bitwise_and(thresh, thresh, mask=mask)



# Make body silhouette mask by inverse of threshold output
cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),
              (0, 0, 255))


body_mask = cv2.bitwise_not(thresh)

# Remove silhouette part from frame by masking it on frame
frame = cv2.bitwise_and(frame, frame, mask=body_mask)

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.imshow("frame", frame)

cv2.waitKey()

# Release the capture
cap.release()
cv2.destroyAllWindows()
