import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors

from collections import defaultdict, Counter

track_history = defaultdict(lambda: [])
model = YOLO("/home/rileyballachay/dev/ultralytics/runs/detect/train31/weights/yolov8_strawberry_mar22_2024.pt")
names = model.model.names

video_path = "IMG_0011.MOV"
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error reading video file"

w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

result = cv2.VideoWriter("object_tracking.avi",
                       cv2.VideoWriter_fourcc(*'mp4v'),
                       fps,
                       (w, h))

# Initialize a counter for all detected classes
class_counts = {'keys':[],'counts':{0:0,1:0,2:0}}

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=False)
        boxes = results[0].boxes.xyxy.cpu()

        if results[0].boxes.id is not None:

            # Extract prediction results
            clss = results[0].boxes.cls.cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            # Update the class counts
            id_cls = list(zip(track_ids,clss))
            for id, cls in id_cls:
                if id not in class_counts['keys']:
                    class_counts['keys'].append(id)
                    class_counts['counts'][int(cls)]+=1

            # Annotator Init
            annotator = Annotator(frame, line_width=2)

            for box, cls, track_id in zip(boxes, clss, track_ids):
                annotator.box_label(box, color=colors(int(cls), True), label=names[int(cls)])

                # Store tracking history
                track = track_history[track_id]
                track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))

                if len(track) > 30:
                    track.pop(0)


                # Plot tracks
                points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                cv2.circle(frame, (track[-1]), 7, colors(int(cls), True), -1)
                frame=cv2.putText(frame,f"Ripe: {class_counts['counts'][2]}, green:{class_counts['counts'][1]}, flowers: {class_counts['counts'][0]}", 
                (10,1000), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),3,1)
                cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)

        result.write(frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

result.release()
cap.release()
cv2.destroyAllWindows()