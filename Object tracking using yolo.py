import cv2
import numpy as np

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load the class names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Set the input and output layers
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Set up the object tracker
tracker = cv2.TrackerKCF_create()

# Capture the video stream
cap = cv2.VideoCapture(0)

# Initialize the object location
bbox = None

while True:
    # Read the next frame
    ret, frame = cap.read()

    # Preprocess the frame
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Feed the frame to the network
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Process the output
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] == 'robot':
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maximum suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw the detected boxes on the frame
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Initialize the object tracker if not already done
        if bbox is None:
            bbox = (x, y, w, h)
            tracker.init(frame, bbox)
        else:
            # Update the object tracker
            ok, bbox = tracker.update(frame)
            if ok:
                # Draw the tracked object
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
            else:
                bbox = None

    # Show the processed frame
    cv2.imshow("Object Detection and Tracking", frame)

    # Wait for the user to quit the program
    if cv2.waitKey(1) == ord('q'):
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
