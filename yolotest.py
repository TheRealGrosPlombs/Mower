from ultralytics import YOLO
import cv2

# Load a model
print("Loading model...\r\n")
model = YOLO("yolo11n.pt")  # pretrained YOLO11n model

print("Opening stream...\r\n")
#cap = cv2.VideoCapture('rtsp://thingino:thingino@192.168.0.234:554/ch1')
cap = cv2.VideoCapture('cs.mp4')

print("Begin detecting...\r\n")
while True:
    # Run batched inference on a list of images
    ret, img= cap.read()
    results = model(img, stream=True)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]    
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
        #masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        #obb = result.obb  # Oriented boxes object for OBB outputs
        #result.show()  # display to screen
        #result.save(filename="result.jpg")  # save to disk
        cv2.imshow('cam',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
	
