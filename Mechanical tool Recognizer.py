import os
print(os.listdir("/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training"))
# 1 ) Importing Various Modules.

# Ignore  the warnings
import warnings
import tensorflow as tf
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
sns.set(style='whitegrid', color_codes=True)

# model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

# preprocess.
from keras.preprocessing.image import ImageDataGenerator

# dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
# from keras.utils import to_categorical
from tensorflow.keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization


import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image

# 2 ) Preparing the Data
# 2.1) Making the functions to get the training and validation set from the Images
X=[]
Z=[]
IMG_SIZE=150
tools_Gasoline_Can_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Gasoline Can'
tools_Hammer_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Hammer'
tools_pebbel_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/pebbel'
tools_Pliers_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Pliers'
tools_Rope_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Rope'
tools_Screw_Driver_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Screw Driver'
tools_Tool_box_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Tool box'
tools_Wrench_DIR='/home/anik/PycharmProjects/Digit_Recognizer_MLP/Training/Wrench'

def assign_label(img,tool_type):
    return tool_type


def make_train_data(tool_type, DIR):
    for img in tqdm(os.listdir(DIR)):
        label = assign_label(img, tool_type)
        path = os.path.join(DIR, img)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))

make_train_data('Gasoline Can',tools_Gasoline_Can_DIR)
print(len(X))

make_train_data('Hammer',tools_Hammer_DIR)
print(len(X))

make_train_data('pebbel',tools_pebbel_DIR)
print(len(X))

make_train_data('Pliers',tools_Pliers_DIR)
print(len(X))

make_train_data('Rope',tools_Rope_DIR)
print(len(X))

make_train_data('Screw Driver',tools_Screw_Driver_DIR)
print(len(X))

make_train_data('Tool box',tools_Tool_box_DIR)
print(len(X))

make_train_data('Wrench',tools_Wrench_DIR)
print(len(X))
# 2.2 ) Visualizing some Random Images

fig, ax = plt.subplots(5, 2)
fig.set_size_inches(15, 15)
for i in range(5):
    for j in range(2):
        l = rn.randint(0, len(Z))
        ax[i, j].imshow(X[l])
        ax[i, j].set_title('Tool: ' + Z[l])

plt.tight_layout()

# 2.3 ) Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding

le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,8)
X=np.array(X)
X=X/255
# 2.4 ) Splitting into Training and Validation Sets

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)

print(y_test)
print(y_test.shape)
print(x_test)
print(x_test.shape)
# 2.5 ) Setting the Random Seeds

np.random.seed(42)
rn.seed(42)
# tf.set_random_seed(42)
tf.random.set_seed(42)

# 3 ) Modelling
# 3.1 ) Building the ConvNet Model

# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='Same', activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(32,(3,3),activation="relu",padding="same"))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(8, activation="softmax"))

# 3.2 ) Using a LR Annealer

batch_size=64
epochs=1

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.5)

# 3.3 ) Data Augmentation to prevent Overfitting

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

# 3.4 ) Compiling the Keras Model & Summary

model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

model.summary()

# 3.5 ) Fitting on the Training set and making predcitons on the Validation set¶


# # model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

# # batch_size=32
# # epochs=64
# # history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),
# #                               epochs = epochs,
# #                               validation_data = (X_test,y_test),
# #                               verbose = 1)

# model.compile(optimizer=Adam(lr=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
# History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
#                               epochs = epochs, validation_data = (x_test,y_test),
#                               verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)

History = model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))

# 4 ) Evaluating the Model Performance¶

plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()



plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()

# 5 ) Visualizing Predictons on the Validation Set
# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)
print(pred_digits)
print(pred_digits.shape)

# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

print(prop_class)
i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break


# CORRECTLY CLASSIFIED FLOWER IMAGES¶
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        #ax[i,j].set_title("Predicted tool : "+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual tool : "+str(le.inverse_transform(np.argmax([y_test[prop_class[count]]]))))
        ax[i,j].set_title("Predicted tool :"+str(le.inverse_transform([pred_digits[prop_class[count]]]))+"\n"+"Actual tool : "+str(le.inverse_transform([np.argmax(y_test[prop_class[count]])])))
        plt.tight_layout()
        count+=1

# MISCLASSIFIED IMAGES OF FLOWERS
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        #ax[i,j].set_title("Predicted Flower : "+str(le.inverse_transform([pred_digits[mis_class[count]]]))+"\n"+"Actual Flower : "+str(le.inverse_transform(np.argmax([y_test[mis_class[count]]]))))
        ax[i,j].set_title("Predicted tool :"+str(le.inverse_transform([pred_digits[prop_class[count]]])))#+"\n"+"Actual tool : "+str(le.inverse_transform([np.argmax(y_test[prop_class[count]])])))
        plt.tight_layout()
        count+=1

model.save("machinetools.h5")