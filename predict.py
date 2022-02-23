import cv2 as cv
import numpy as np
import os

if __name__ == "__main__":

    # Initialize required parameters for yolo model
    INPUT_SIZE = (416, 416)
    SCALING_FACTOR = 1/255
    MEAN_SUBTRACTION = (0, 0, 0)
    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4


    # Select the image to make prediction on
    img = cv.imread("images/4.jpg")
    img_height, img_width = img.shape[:2]


    # Load the yolo model using pretrained weights and model configurations.
    model = cv.dnn.readNet('yolov3.weights', 'yolov3.cfg')

    
    # Get those layers which are giving the output.
    # In yolo v3 model, there are 3 such output layers. 
    # These 3 output layers help in easily detecting objects of small and large sizes. 
    outputLayers = model.getUnconnectedOutLayersNames()


    # Using the yolo model trained on COCO dataset.
    # It contains 80 classes of objects.
    classes = []
    with open('coco_names.txt', 'r') as name_file:
        classes = name_file.read().splitlines()


    # Change the image to a format which is required as yolo model input.
    # blob is a (1, 416, 416, 3) shape tensor.
    # 1 is the number of images (since we are passing only one image at a time)
    # 416 x 416 is the input shape
    # 3 is the number of chanels
    blob = cv.dnn.blobFromImage(img, SCALING_FACTOR, INPUT_SIZE, MEAN_SUBTRACTION, swapRB=True, crop=False)


    # Provide input and give it a forward pass through the pretrained yolo model.
    # all_outputs contains the output from all 3 output layers of the model.
    model.setInput(blob)
    all_outputs = model.forward(outputLayers)


    # Store the required information to detect an object in these Lists.
    # bboxes contain the co-ordinates of key points of bounding box.
    # class_ids contains the id of the classs which are detected for various objects in the image. 
    # class_confidence contains the confidence value with which the model has detected a class.
    bboxes = []
    class_ids = []
    class_confidences = []

    for output in all_outputs:
        for detection in output:
            box_score = detection[4]
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)

            if class_scores[class_id] > CONFIDENCE_THRESHOLD:
                cx = int(detection[0] * img_width)
                cy = int(detection[1] * img_height)
                box_width = int(detection[2] * img_width)
                box_height = int(detection[3] * img_height)

                x = int(cx - box_width/2)
                y = int(cy - box_height/2)

                bboxes.append([x, y, box_width, box_height])
                class_ids.append(class_id)
                class_confidences.append(class_scores[class_id])


    # Since the yolo model can give multiple bounding boxes for the same object, 
    # we apply non-max suppression on the bounding boxes and keep only those boxes
    # whose IOU (Intersection over Union) value is less than a certain threshold.
    # IOU is a measure of overlap of bounding boxes on each other.
    final_indices = cv.dnn.NMSBoxes(bboxes, class_confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)


    # Finally draw the final bounding boxes around detected objects.
    for i in final_indices:
        box = bboxes[i]
        object_class = classes[class_ids[i]]
        confidence = class_confidences[i] * 100
        confidence = round(confidence, 1)

        print(confidence)

        x, y, w, h = box[:]
        
        cv.putText(img, f'{object_class} {confidence}', (x, y-5), cv.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
        
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)


    cv.imshow("img", img)

    cv.waitKey(0)
    cv.destroyAllWindows()