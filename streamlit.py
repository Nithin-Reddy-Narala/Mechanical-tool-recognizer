from tkinter import *
from PIL import ImageGrab
import imageio
import tkinter.font as font
import numpy as np
import tensorflow as tf
from PIL import Image
image = Image.open('nn.jpg')


model = tf.keras.models.load_model('latestmodel.h5')

import streamlit as st
st.title("Mechanical   Tools   Image   Recognizer")
st.image(image, caption=None, width=None, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
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
            #st.write(class_list[i])
            if class_list[i] == 'Gasoline Can':
                st.header("Gasoline Can")
                st.write('''It is a fuel container such as a steel can, bottle, drum, etc. for transporting, storing, and dispensing various fuels. 
                
How to use? watch this video -> https://www.youtube.com/watch?v=-uklKn0mVek''')
            elif class_list[i] == 'Hammer':
                st.header("Hammer")
                st.write('''A hammer is a tool, most often a hand tool, consisting of a weighted head fixed to a long handle that is swung to deliver an impact to a small area of an object. This can be, for example, to drive nails into wood, to shape metal (as with a forge), or to crush rock. Hammers are used for a wide range of driving, shaping, breaking and non-destructive striking applications.
                
How to use? watch this video -> https://www.youtube.com/watch?v=g6dvj4MKjDc''')
            elif class_list[i] == 'Pebbel':
                st.header("Pebbel")
                st.write("A pebble is a clast of rock with a particle size of 4–64 mm (0.16–2.52 in) based on the Udden-Wentworth scale of sedimentology. Pebbles are generally considered larger than granules (2–4 mm (0.079–0.157 in) in diameter) and smaller than cobbles (64–256 mm (2.5–10.1 in) in diameter).")
            elif class_list[i] == 'Pliers':
                st.header("Pliers")
                st.write('''Pliers are a hand tool used to hold objects firmly, possibly developed from tongs used to handle hot metal in Bronze Age Europe. They are also useful for bending and compressing a wide range of materials. Generally, pliers consist of a pair of metal first-class levers joined at a fulcrum positioned closer to one end of the levers, creating short jaws on one side of the fulcrum, and longer handles on the other side. This arrangement creates a mechanical advantage, allowing the force of the hand's grip to be amplified and focused on an object with precision. The jaws can also be used to manipulate objects too small or unwieldy to be manipulated with the fingers.
                
How to use? you can watch this video -> https://www.youtube.com/watch?v=VF3xOXZn8WU''')
            elif class_list[i] == 'Rope':
                st.header('Rope')
                st.write('''A rope is a group of yarns, plies, fibers or strands that are twisted or braided together into a larger and stronger form. Ropes have tensile strength and so can be used for dragging and lifting. Rope is thicker and stronger than similarly constructed cord, string, and twine.''')
            elif class_list[i] == 'Screw Driver':
                st.header('Screw Driver')
                st.write('''A screwdriver is a tool, manual or powered, used for driving screws. A typical simple screwdriver has a handle and a shaft, ending in a tip the user puts into the screw head before turning the handle. This form of the screwdriver has been replaced in many workplaces and homes with a more modern and versatile tool, a power drill, as they are quicker, easier, and also can drill holes.
                
How to use? you can watch this video ->https://www.youtube.com/watch?v=E-wKHdo4b0w''')
            elif class_list[i] == 'Tool Box':
                st.header('Tool Box')
                st.write('''Toolboxes are an ideal piece of kit for keeping tools organised and making them easy to transport and store. With the box divided into sections for smaller and larger parts, it’s easy to access the right equipment at the right time. Tool boxes are available in range of sizes, perfect for both domestic and commercial users.
                
How to use? you can watch this video -> https://www.youtube.com/watch?v=4T3UvYN-bbw''')
            elif class_list[i] == 'Wrench':
                st.header('Wrench')
                st.write('''A wrench or spanner is a tool used to provide grip and mechanical advantage in applying torque to turn objects—usually rotary fasteners, such as nuts and bolts—or keep them from turning.In the UK, Ireland, Australia, and New Zealand spanner is the standard term. The most common shapes are called open-ended spanner and ring spanner. The term wrench is generally used for tools that turn non-fastening devices (e.g. tap wrench and pipe wrench), or may be used for a monkey wrench—an adjustable pipe wrench.
                
How to use? you can watch this video ->https://www.youtube.com/watch?v=latrFJ7uFiM''')
            break







    
