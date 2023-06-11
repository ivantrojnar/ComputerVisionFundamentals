# Computer Vision Fundamentals

This project aims to train a deep learning model to perform object detection on images using the COCO dataset format. The trained model can then be used to detect objects in new images and provide confidence levels for the detected objects.

## Project Structure

The project structure is as follows:
`.
├── LI13

│   ├── images

│   ├── result.json

├── main.py

└── detection.py
`

- The `LI13` directory contains the images folder and `result.json` file, which were generated in Label Studio when exporting the project in COCO format.
- The `main.py` file is responsible for training the object detection model from scratch.
- The `detection.py` file provides functionality to load the trained model and perform object detection on input images.
 
