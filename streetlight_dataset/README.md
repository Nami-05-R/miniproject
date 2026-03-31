Streetlight dataset notes

The dataset currently added to this project is a YOLO detection export, not an
ON/OFF/DIM classification dataset.

That means the training flow is:
1. train a streetlight detector on the labeled lamp locations
2. use brightness analysis inside detected streetlight regions to infer
   `ON`, `DIM`, or `OFF`

You can either:
- keep the zip file `Street Light.v1i.yolov8.zip` in the project root and let
  `streetlight_train.py` extract it automatically, or
- manually extract it into this folder so it contains:

streetlight_dataset/
train/
images/
labels/
valid/
images/
labels/
test/
images/
labels/
data.yaml

The app will automatically pick up the trained detector weights from:
`runs/detect/streetlight_det/weights/best.pt`
