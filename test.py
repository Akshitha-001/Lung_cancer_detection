import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("lung_cancer_cnn_model.h5")

img = cv2.imread("test_image.jpg")
img = cv2.resize(img, (128,128))
img = img / 255.0
img = np.reshape(img, (1,128,128,3))

prediction = model.predict(img)

if prediction > 0.5:
    print("Prediction: Malignant (Cancer Detected)")
else:
    print("Prediction: Benign (No Cancer)")
