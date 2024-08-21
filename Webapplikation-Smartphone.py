import streamlit as st 
import keras 
import tensorflow
import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt



CATEGORIES = ["semi-finished product", "finished product"]

st.title("Please take a picture of the Fischertechnik vehicle!")

model = keras.saving.load_model("image_classifier_custom_binary_3.keras")
img_data = st.camera_input(label='load image')

if img_data: 
    st.image(img_data)
    IMG_SIZE = 100
    img_data = Image.open(img_data)
    img_data = img_data.save("img.jpg")
    img_data = cv.imread("img.jpg" ,cv.IMREAD_GRAYSCALE)
    img_data =cv.resize(img_data, (IMG_SIZE,IMG_SIZE))
    img_data = img_data/255.0
    new_image=np.array(img_data).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    #st.image(img_data)
    prediction = model.predict([new_image])
    st.title(int(prediction[0][0]))
    st.title(CATEGORIES[int(prediction[0][0])])