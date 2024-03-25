from ultralytics import YOLO
import cv2
import numpy as np


# Load a model
model = YOLO("/home/rileyballachay/dev/ultralytics/runs/detect/train31/weights/yolov8_strawberry_mar22_2024.pt", task='eval')  # load a pretrained model (recommended for training)

model_old = YOLO("/home/rileyballachay/dev/ultralytics/runs/detect/train26/weights/strawberry_model_sep_2023.pt", task='eval')  # load a pretrained model (recommended for training)

cap = cv2.VideoCapture('IMG_0011.MOV')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

# we want a frame every fps frames
fps = round(cap.get(cv2.CAP_PROP_FPS))//3

frame_buffer = []

i=0
# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret==True:
        break

    if not (i%fps):

        # Display the resulting frame
        frame_buffer.append(frame)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    i+=1



font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,1000)
fontScale              = 3
fontColor              = (0,255,0)
thickness              = 3
lineType               = 2

video_results = []

for frame in frame_buffer:

    results = model(frame)[0].plot(labels=0)
    results_old = model_old(frame)[0].plot(labels=0)

    results_old=cv2.putText(results_old,'Model v1', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)

    results=cv2.putText(results,'Model v2', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)


    concat_results = np.concatenate([results_old, results],axis=1)

    video_results.append(concat_results)




out = cv2.VideoWriter('mar25_model_results_IMG_0011.avi', cv2.VideoWriter_fourcc(*"XVID"), 2.0, video_results[0].shape[:2][::-1])

for frame in video_results:
    out.write(frame)

out.release()
    

raise Exception(len(frame_buffer))