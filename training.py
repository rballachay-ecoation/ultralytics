from ultralytics import YOLO


# Load a model
model = YOLO("/home/rileyballachay/dev/ultralytics/runs/detect/train31/weights/yolov8_strawberry_mar22_2024.pt", task='eval')  # load a pretrained model (recommended for training)

# Use the model
#model.train(data="/home/rileyballachay/dev/ultralytics/strawberry.yaml", epochs=100)  # train the model
#metrics = model.val()  # evaluate model performance on the validation set
#results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

#model = YOLO('/home/rileyballachay/dev/ultralytics/runs/detect/train9/weights/best.pt', task='detect')

#model('/home/rileyballachay/dev/ultralytics/strawberry_data/images/train/0-R0010007.png')
path = model.export(format="coreml",nms=True)  # export the model to ONNX format