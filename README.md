<!-- hide -->
# Image Classifier Project Tutorial
<!-- endhide -->

- You will write an algorithm to classify whether images contain either a dog or a cat.  This is easy for humans, dogs, and cats. Your computer will find it a bit more difficult.

>Don't forget to always be resourceful!

## 🌱  How to start this project

You will not be forking this time, please take some time to read this instructions:

1. Create a new repository based on [machine learning project](https://github.com/4GeeksAcademy/machine-learning-python-template/generate) by [clicking here](https://github.com/4GeeksAcademy/machine-learning-python-template).
2. Open the recently created repostiroy on Gitpod by using the [Gitpod button extension](https://www.gitpod.io/docs/browser-extension/).
3. Once Gitpod VSCode has finished opening you start your project following the Instructions below.

## 🚛 How to deliver this project

Once you are finished creating your image classifier, make sure to commit your changes, push to your repository and go to 4Geeks.com to upload the repository link.

## 📝 Instructions

**Image Classifier**

The dataset is comprised of photos of dogs and cats provided as a subset of photos from a much larger dataset of 3 million manually annotated photos. The dataset was developed as a partnership between Petfinder.com and Microsoft.

The dataset was originally used as a CAPTCHA, that is, a task that it is believed a human finds trivial, but cannot be solved by a machine, used on websites to distinguish between human users and bots. The task was referred to as "Asirra". When "Asirra" was presented, it was mentioned 'that user studies indicate it can be solved by humans 99.6% of the time in under 30 seconds. Barring a major advance in machine vision, we expect computers will have no better than a 1/54,000 chance of solving it'.

At the time that the competition was posted, the state-of-the-art result was achieved with an SVM and described in a 2007 paper with the title “Machine Learning Attacks Against the Asirra CAPTCHA” (PDF) that achieved 80% classification accuracy. It was this paper that demonstrated that the task was no longer a suitable task for a CAPTCHA soon after the task was proposed.

The dataset is straightforward to understand and small enough to fit into memory and get started with computer vision and convolutional neural networks.

Dataset links:

https://www.kaggle.com/c/dogs-vs-cats/data

**Step 1:**

Download the datatset folder and unzip files. You will now have a folder called ‘train/‘ that contains 25,000 .jpg files of dogs and cats. The photos are labeled by their filename, with the word “dog” or “cat“.

**Step 2:**

Import the following libraries:

```py
import keras,os
from keras.models import Sequential  #as all the layers of the model will be arranged in sequence
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator # as it imports data with labels easily into the model. It has functions to rescale, rotate, zoom, etc. This class alters the data on the go while passing it to the model.
import numpy as np
```
**Step 3:**

Load and plot the first nine photos of dogs in a single figure. Repeat the same for cats. You can see that the photos are color and have different shapes and sizes. 

The photos will have to be reshaped prior to modeling so that all images have the same shape. This is often a small square image. Smaller inputs mean a model that is faster to train so we will choose a fixed size of 200×200 pixels.

We could load all of the images, reshape them, and store them as a single NumPy array. This could fit into RAM on many modern machines, but not all, especially if you only have 8 gigabytes to work with.
We can write custom code to load the images into memory and resize them as part of the loading process, then save them ready for modeling.

1. If you have more than 12 gigabytes of RAM use the Keras image processing API to load all 25,000 photos in the training dataset and reshape them to 200×200 square photos. The label should also be determined for each photo based on the filenames. A tuple of photos and labels should be saved.

2. If you do not have more than 12 gigabytes of RAM, load the images progressively using the Keras ImageDataGenerator class and flow_from_directory() API. This will be slower to execute but will run on more machines. This API prefers data to be divided into separate train/ and test/ directories, and under each directory to have a subdirectory for each class. 

**Step 4:**

Create an object of ImageDataGenerator for both training and testing data and pass the folder which has train data to the object trdata and similarly pass the folder which has test data to the object tsdata. 

The ImageDataGenerator will automatically label all the data inside the cat folder as cat and vis-à-vis for dog folder. This way, data is quickly ready to be passed to the neural network.

**Step 5:**

Any classifier fit on this problem will have to be robust because some images show the cat or dog in a corner or maybe 2 cats or dogs in the same photo. VGG16 is a convolution neural net (CNN ) architecture used to win the ILSVR(Imagenet) competition in 2014. It is considered to be one of the excellent vision model architecture till date.

The most unique thing about VGG16 is that instead of having a large number of hyper-parameter they focused on having convolution layers of 3x3 filter with a stride 1 and always used the same padding and maxpool layer of 2x2 filter of stride 2. It follows this arrangement of convolution and max pool layers consistently throughout the whole architecture. In the end, it has 2 FC(fully connected layers) followed by a softmax for output. The 16 in VGG16 refers to it has 16 layers that have weights. This network is a pretty large network and it has about 138 million (approx) parameters.

Initialize the model by specifying that the model is a sequential model. After initialising the model add:

→ 2 x convolution layer of 64 channel of 3x3 kernel and same padding.

→ 1 x maxpool layer of 2x2 pool size and stride 2x2.

→ 2 x convolution layer of 128 channel of 3x3 kernel and same padding.

→ 1 x maxpool layer of 2x2 pool size and stride 2x2.

→ 3 x convolution layer of 256 channel of 3x3 kernel and same padding.

→ 1 x maxpool layer of 2x2 pool size and stride 2x2.

→ 3 x convolution layer of 512 channel of 3x3 kernel and same padding.

→ 1 x maxpool layer of 2x2 pool size and stride 2x2.

→ 3 x convolution layer of 512 channel of 3x3 kernel and same padding.

→ 1 x maxpool layer of 2x2 pool size and stride 2x2.

Add relu(Rectified Linear Unit) activation to each layer so that all the negative values are not passed to the next layer.

Let's see some first rows to have an idea, and continue with all the layers:

```py
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
```

**Step 6:**

After creating all the convolution, pass the data to the dense layer. In order to do that, you should first flatten the vector which comes out of the convolutions and then add:

→ 1 x Dense layer of 4096 units

→ 1 x Dense layer of 4096 units

→ 1 x Dense Softmax layer of 2 units

Use RELU activation for both of the dense layers in order to stop forwarding negative values through the network. Use a 2 unit dense layer in the end with softmax activation as you have 2 classes to predict. The softmax layer will output the value between 0 and 1 based on the confidence of the model that which class the images belong.

**Step 7:**

Import Adam optimizer and use it to compile the model. Specify a learning rate for it.

**Step 8:**

Check the summary of the model

**Step 9:**

Import ModelCheckpoint and EarlyStopping method from keras. Create an object of both and pass that as callback functions to fit_generator.

**Step 10:**

Once you have trained the model, visualize training/validation accuracy and loss. 

**Step 11:**

Load the best saved model and pre-process the image, then pass the image to the model and make predictions.

**Step 12:**

Use your app.py file to create your image classifier. 

In your README file write a brief summary.

Solution guide: 

https://github.com/4GeeksAcademy/image-classifier-project-tutorial/blob/main/solution_guide.ipynb
