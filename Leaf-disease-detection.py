import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2
import random
import os
from os import listdir
from PIL import Image
from sklearn.preprocessing import label_binarize, LabelBinarizer
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dropout, Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json
from tensorflow.keras.utils import to_categorical

#Plotting 12 images to check dataset
#Now we will observe some of the iamges that are their in our dataset. we will plot 12 images here using the matplotlib library.
plt.figure(figsize=(12,12))
path = "C:\\Users\\NITHYASRI\\Desktop\\Plant_images\\Potato_Early_blight"
for i in range(1,17):
    plt.subplot(4,4,1)
    plt.tight_layout()
    rand_img = imread(path +'/'+ random.choice(sorted (os.listdir(path))))
    plt.imshow(rand_img)
    plt.xlabel(rand_img.shape[1], fontsize = 10)#width of image
    plt.ylabel(rand_img.shape[0], fontsize = 10)#height of image

#Converting Images to array 
def convert_image_to_array(image_dir):
    try:
        Image cv2.imread(image_dir)
        if image is not None:
            Image cv2.resize(inage, (256,256))
            Image = cv2.cvtColor(image, cv2.COLOR BGR2GRAY)
            return ing_to_array(image)
        else:
            return np.array([])
    except Exception as e:
        print("Error: {e}")
        return None

dir = "C:\\Users\\NITHYASRI\\Desktop\\Plant_Images"
root_dir = listdir(dir)
image_list, label_list = [],[]
all_labels = ['Corn-Common_rust', 'Potato-Early_blight', 'Tomato-Bacterial_spot']
binary_labels = [0,1,2]
temp = -1
#Reading and converting image to numpy array
#Now we will convert all the images into numpy array.
for directory in root_dir:
    plant_image_list = listdir(f"{dir}/{directory}")
    temp += 1
for files in plant_image_list:
    image_path = f"{dir}/{directory}/{files}"
    image_list.append(convert_image_to_array(image_path))
    label_list.append(binary_labels[temp])

#Visualize the number of classes count 
label_counts = pd.DataFrame(label_list).value_counts() 
label_counts.head()
#it is a balanced dataset as you can see

image_list[0].shape
label_list = np.array(label_list)
label_list.shape

x_train, x_test, y_train, y_test = train_test_split(image_list, label_list, test_size=0.2, randon_state = 10)

#Now we will normalize the dataset of our inages. As pixel values ranges from è to 255 so we will divide each image pixel with 2 
x_train = np.array(x_train, dtype=np.float16) / 225.0
x_test = np.array(x_test, dtype-np.float16) / 225.0
x_trainx_train.reshape(-1, 256,256,3) 
x_test = x_test.reshape(-1, 256,256,3)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(256,256,3), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer = Adam (0.0001), metrics=['accuracy'])

#Next we will split the dataset into validation and training data.
# Splitting the training data set into training and validation data sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.2)

epochs = 50
batch_size = 128
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

#plot the training history
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'val'])
plt.show()

print("[INFO] Calculating model accuracy")
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")

y_pred = model.predict(x_test)
#plotting image to compare
img = array_to_img(x_test[10])
img


#finding max value from prediction list and comparing original value vs predicted
print("Originaly: ", all_labels[np.atgmax(y_test[11])])
print("Predicted: ", all_labels[np.argmax(y_pred[11])])

if all_labels[np.argmax(y_pred[10])] == all_labels[np.argmax(y_test[10])]:
    {
        print("Correctly predicted")
    }
else:
    {
        print("Incorrectly predicted")
    }

if all_labels[np.argmax(y_pred[1])] == all_labels[np.argmax(y_test[0])]:
    {
        print("Correctly predicted")
    }
else:
    {
        print("Incorrectly predicted")
    }
