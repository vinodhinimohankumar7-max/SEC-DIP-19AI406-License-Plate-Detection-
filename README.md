# SEC-DIP-19AI406-License-Plate-Detection-
License Plate Detection using OpenCV and Haar Cascade Classifier
Name: vinodhini
Register Number: 212225230305
## Aim
To implement a License Plate Detection system using OpenCV and Haar Cascade Classifier, draw bounding boxes, crop the detected region, and blur the license plate to improve privacy. The detection accuracy is improved by tuning Haar Cascade parameters.

## Software Used
Python 3.7 or above
OpenCV (opencv-python)
NumPy
Matplotlib
Jupyter Notebook (Anaconda)
Haar Cascade File: haarcascade_russian_plate_number.xml

## Algorithm
Import necessary libraries such as OpenCV and Matplotlib
Read the input vehicle image
Convert the original image to grayscale for faster computation
Load the Haar Cascade classifier for license plate detection
Detect license plate using detectMultiScale function
Draw rectangle around detected area
Crop the detected region using numpy slicing with (x, y, w, h) values
Apply median blurring on the cropped region
Replace the original region with blurred version
Display final result using Matplotlib

 ## Program
```
import cv2
import matplotlib.pyplot as plt

def display(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,6))
    plt.imshow(img_rgb)
    plt.axis('off')

img = cv2.imread("DATA/car_plate.jpg")
plate_cascade = cv2.CascadeClassifier("DATA/haarcascades/haarcascade_russian_plate_number.xml")

def detect_and_blur_plate(img):
    img_copy = img.copy()
    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]
        blurred_roi = cv2.medianBlur(roi, 15)
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

result = detect_and_blur_plate(img)
display(result)
```
## Output

![alt text](<Screenshot 2026-09-17 210745.png>)

![alt text](<Screenshot 2026-09-17 210802.png>)

![alt text](<Screenshot 2026-09-17 210818.png>)
