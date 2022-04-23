# Automatic License Plate Recognition (ALPR)

## Introduction
- This code takes a two step approach 
  - License plates are first detected using [YOLOv4](https://github.com/AlexeyAB/darknet)
  - OCR is then applied using [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) on the detected license plates
- Reference : https://learnopencv.com/automatic-license-plate-recognition-using-deep-learning/

## Notebooks

- **[alpr_example_darknet_paddleocr.ipynb](https://github.com/wayne0git/ml_cv_basics/blob/master/alpr/alpr_example_darknet_paddleocr.ipynb)**
  - This notebook contains the pipeline required for end to end inference of the Automatic License plate recognition on images and videos along with the implementation of tracker

- **[license_plate_detection_train_yolov4.ipynb](https://github.com/wayne0git/ml_cv_basics/blob/master/alpr/license_plate_detection_train_yolov4.ipynb)**
  - This notebook contains end to end implementation of license plate detection using YOLOv4
