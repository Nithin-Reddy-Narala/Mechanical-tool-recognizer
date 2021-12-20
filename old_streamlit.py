from tkinter import *
from PIL import ImageGrab
import imageio
import tkinter.font as font
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('machinetools1.h5')

import streamlit as st
st.header("Machine Tools Image Detection")

file = st.file_uploader("Please upload an image file", type=["jpg", "png",'Jpeg'])

import cv2
from PIL import Image, ImageOps
import numpy as np

class_list = ['Gasoline Can','Hammer','Pebbel','Pliers','Rope','Screw Driver','Tool Box','Wrench']
def import_and_predict(image_data, model):
    size = (75, 75)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resize = (cv2.resize(img, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)) / 255.

    img_reshape = img_resize[np.newaxis, ...]

    prediction = model.predict(img_reshape)

    return prediction


if file is None:

    st.text("Please upload an image file")
else:

    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)

    for i in range(0,len(class_list)):
        if np.argmax(prediction) == i:
            st.write(class_list[i])
            break







    
