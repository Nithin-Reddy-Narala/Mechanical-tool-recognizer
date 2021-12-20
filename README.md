# Mechanical-tool-recognizer

![This is an image](https://github.com/Nithin-Reddy-Narala/Mechanical-tool-recognizer/blob/main/Image.jpg)

This project is a part of Data Analytics course.

## Project Description

* Main scope of this project is to identify different type of mechanical tools and show how to use it in a web-interface. 
* A Jupyter notebook where a CNN Neural Network model has been compiled, fit & trained using 6k images of 8 different tools (Hammer, Screw driver, Wrench..etc).
* A python script (streanlit.py exectuable) which serve as a GUI (using streamlit.io) to use the model to recognize new & unseen tool data.

## Technologies used

* The main technologies used here are Python, using the libraries such as Tensorflow, Keras (for the AI model), Pytorch, opencv, pillow as well as stremalit to build a simple web-interface for GUI.

## Challenges

The major challenges during this project:

* Finding a dataset of images that was large enough ( > few thousand different images), split between categories of different classes, that was also pretty balanced.
* Constructing a model in terms of layer number, layer type, parameters (learning rate, epochs, number of steps, etc). to maximize the model accuracy.
* Shifting from the Jupyter environment to streamlit GUI.

## How to use & install

The project is straightforward to install, use & modify:

* Original data set can be downloaded from Kaggle https://www.kaggle.com/salmaneunus/mechanical-tools-dataset
* Run the Jupyter notebook (mechanical tool recognizer.ipynb) in your preferred Python IDE.
* Run the Jupyter streamlit.py from your terminal to see the streamlit GUI.

Nithin Reddy Narala




